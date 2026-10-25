"""Part 2 loop iteration: baseline prior x level + credibility-weighted gender factor (motor, contents)
   + a small, heavily shrunk new-business severity adjustment for motor only
   + a heavily shrunk per-peril level factor for buildings
   + a heavily shrunk weather adjustment (hail_days, wind_exposure_index) for storm and fire in buildings.
fit_predict(train, valid, line):
  train: features + prior_* columns + exposure, year, policy_id + judge columns (target_b, ...) for the fit rows;
  valid: same without judge columns. Return the predicted cost for valid (same units as target_b).
"""
import numpy as np
import pandas as pd


def fit_predict(train, valid, line):
    p_tr = train["prior_mu_src"].to_numpy(float)
    p_va = valid["prior_mu_src"].to_numpy(float)
    y = train["target_b"].to_numpy(float)
    level = y.sum() / p_tr.sum()
    pred = p_va * level

    # Gender credibility factor for motor and contents
    if line in ("motor", "contents"):
        k = 2000.0 * y.mean()  # credibility constant: 2000 rows of average cost
        g_tr = train["gender"].astype(str).to_numpy()
        g_va = valid["gender"].astype(str).to_numpy()
        fac = {}
        for g in np.unique(g_tr):
            m = g_tr == g
            fac[g] = (y[m].sum() + k) / (p_tr[m].sum() * level + k)
        pred = pred * np.array([fac.get(g, 1.0) for g in g_va])

    # New-business severity adjustment for motor only
    if line == "motor":
        nb_tr = train["new_business"].astype(bool).to_numpy()
        nb_va = valid["new_business"].astype(bool).to_numpy()
        n_nb = nb_tr.sum()
        if n_nb > 0:
            k_nb = 5000.0 * y.mean()  # very strong shrinkage
            nb_cost_tr = y[nb_tr].sum()
            nb_prior_tr = p_tr[nb_tr].sum() * level
            nb_factor = (nb_cost_tr + k_nb) / (nb_prior_tr + k_nb)
            adj = np.where(nb_va, nb_factor, 1.0)
            pred = pred * adj

    # Per-peril level factors for buildings
    if line == "buildings":
        peril_cols_f = [c for c in train.columns if c.startswith("prior_f_")]
        peril_cols_s = [c for c in train.columns if c.startswith("prior_s_")]
        
        if len(peril_cols_f) > 0:
            peril_factors = {}
            for pf, ps in zip(peril_cols_f, peril_cols_s):
                peril_name = pf.replace("prior_f_", "")
                if ps not in train.columns:
                    continue
                
                f_tr = train[pf].to_numpy(float)
                s_tr = train[ps].to_numpy(float)
                peril_cost_tr = f_tr * s_tr
                
                denom = np.where(p_tr > 0, p_tr, 1.0)
                w_tr = peril_cost_tr / denom
                
                actual_cost_p = (y * w_tr).sum()
                prior_cost_p = peril_cost_tr.sum()
                
                k_p_cost = 1000.0 * y.mean()
                if prior_cost_p > 0:
                    level_p = (actual_cost_p + k_p_cost) / (prior_cost_p + k_p_cost)
                    peril_factors[peril_name] = level_p
                else:
                    peril_factors[peril_name] = 1.0
            
            valid_adjustment = np.zeros(len(valid))
            for pf, ps in zip(peril_cols_f, peril_cols_s):
                peril_name = pf.replace("prior_f_", "")
                if peril_name not in peril_factors:
                    continue
                
                f_va = valid[pf].to_numpy(float)
                s_va = valid[ps].to_numpy(float)
                peril_cost_va = f_va * s_va
                
                denom_va = np.where(p_va > 0, p_va, 1.0)
                w_va = peril_cost_va / denom_va
                
                valid_adjustment += w_va * peril_factors[peril_name]
            
            pred = pred * valid_adjustment

            # Weather adjustment for storm and fire in buildings
            # Check if weather columns exist
            if "hail_days" in train.columns and "wind_exposure_index" in train.columns:
                hail_tr = train["hail_days"].to_numpy(float)
                wind_tr = train["wind_exposure_index"].to_numpy(float)
                hail_va = valid["hail_days"].to_numpy(float)
                wind_va = valid["wind_exposure_index"].to_numpy(float)
                
                # Standardize weather variables for stable estimation
                hail_mean_tr = hail_tr.mean()
                hail_std_tr = max(hail_tr.std(), 1e-8)
                wind_mean_tr = wind_tr.mean()
                wind_std_tr = max(wind_tr.std(), 1e-8)
                
                hail_z_tr = (hail_tr - hail_mean_tr) / hail_std_tr
                wind_z_tr = (wind_tr - wind_mean_tr) / wind_std_tr
                
                # For each weather-sensitive peril (storm, fire), estimate a linear weather effect
                # The weather effect multiplies the peril's cost component
                # We estimate the slope by regressing the ratio of actual to prior cost (weighted by peril weight) on weather z-scores
                # But since we only have total cost, we use the peril's weight to attribute cost
                # Simpler: estimate a single weather slope for storm and fire combined, or per peril
                
                weather_factors = {}
                for peril_name in ["storm", "fire"]:
                    pf = f"prior_f_{peril_name}"
                    ps = f"prior_s_{peril_name}"
                    if pf not in train.columns or ps not in train.columns:
                        continue
                    
                    f_tr_p = train[pf].to_numpy(float)
                    s_tr_p = train[ps].to_numpy(float)
                    peril_cost_tr_p = f_tr_p * s_tr_p
                    denom_p = np.where(p_tr > 0, p_tr, 1.0)
                    w_tr_p = peril_cost_tr_p / denom_p
                    
                    # Attribute actual cost to this peril
                    actual_p = (y * w_tr_p).sum()
                    prior_p = peril_cost_tr_p.sum()
                    
                    if prior_p <= 0:
                        continue
                    
                    # Estimate weather effect: 
                    # The ratio actual/prior for rows weighted by w_tr_p
                    # We want to capture how this ratio varies with weather
                    # Use a simple weighted least squares: weight = w_tr_p * prior_p (proportional to prior cost)
                    # y_ratio = y / (p_tr * level) but we want peril-specific
                    # Actually, let's estimate the slope of (y * w_tr_p) / (p_tr * w_tr_p) on weather
                    # This is equivalent to estimating the slope of y/p_tr on weather, weighted by w_tr_p
                    # Let's compute the ratio for each row: r_i = y_i / (p_tr_i * level)
                    # Then the peril-specific ratio is r_i * (w_tr_p_i / w_tr_p_i) = r_i, but weighted by w_tr_p
                    # This is getting complicated. Let's use a simpler approach:
                    # Estimate the weather slope by regressing y on weather, using the prior as an offset
                    # But we want to isolate the peril-specific weather effect.
                    
                    # Alternative: estimate the weather effect on the peril's cost component directly
                    # peril_cost_actual_i = y_i * w_tr_p_i
                    # peril_cost_prior_i = p_tr_i * w_tr_p_i = peril_cost_tr_p_i
                    # ratio_i = peril_cost_actual_i / peril_cost_prior_i = y_i / p_tr_i
                    # So the ratio is the same as the overall ratio! The weather effect is not peril-specific in this formulation.
                    
                    # Let's try a different approach: estimate the weather effect on the overall cost, then apply it to storm and fire
                    # This is what the current best does for level, but we want weather-specific
                    # Estimate slope_hail and slope_wind from y vs weather, using prior as offset
                    # y_i = p_tr_i * level * (1 + alpha * hail_z_i + beta * wind_z_i)
                    # This is a multiplicative model. Let's use log or just linear in the ratio
                    # ratio_i = y_i / (p_tr_i * level)
                    # ratio_i = 1 + alpha * hail_z_i + beta * wind_z_i
                    # Estimate alpha and beta by OLS on ratio_i ~ hail_z_i + wind_z_i
                    # But this estimates the overall weather effect, not peril-specific
                    # For now, let's assume the weather effect is common across perils, but we only apply it to storm and fire
                    # This is a simplification, but it's better than nothing
                    
                    # Actually, let's estimate the weather effect per peril by using the peril's weight in the regression
                    # The contribution of peril p to the total cost is w_tr_p_i * p_tr_i
                    # The weather effect on peril p's cost is what we want
                    # Let's estimate: w_tr_p_i * p_tr_i * (1 + alpha_p * hail_z_i + beta_p * wind_z_i)
                    # Sum over perils = p_tr_i * (1 + sum_p w_tr_p_i * alpha_p * hail_z_i + sum_p w_tr_p_i * beta_p * wind_z_i)
                    # This is a linear model: y_i = p_tr_i + p_tr_i * (A * hail_z_i + B * wind_z_i)
                    # where A = sum_p w_tr_p_i * alpha_p, B = sum_p w_tr_p_i * beta_p
                    # We can estimate A and B, then assume they are proportional to the peril's weight
                    # alpha_p = A * w_tr_p_i / sum_q w_tr_q_i, but this is row-dependent
                    # Let's just estimate A and B and apply them to storm and fire with their average weights
                    pass
                
                # Estimate overall weather effect
                ratio_tr = y / (p_tr * level)
                # OLS: ratio_tr ~ hail_z_tr + wind_z_tr
                X = np.column_stack([np.ones(len(hail_z_tr)), hail_z_tr, wind_z_tr])
                # Weighted least squares: weight = p_tr (prior cost)
                W = p_tr
                XW = X * np.sqrt(W)[:, None]
                yW = ratio_tr * np.sqrt(W)
                try:
                    beta_ols, _, _, _ = np.linalg.lstsq(XW, yW, rcond=None)
                    intercept, alpha_hail, beta_wind = beta_ols
                except:
                    alpha_hail, beta_wind = 0.0, 0.0
                
                # Shrinkage: if the estimated slopes are small, shrink toward 0
                # Use a simple shrinkage: if |alpha| < 0.05, set to 0; else keep
                # Better: shrink based on the standard error
                # For now, use a fixed shrinkage factor
                shrink_factor = 0.5  # heavy shrinkage
                alpha_hail_shrunk = alpha_hail * shrink_factor
                beta_wind_shrunk = beta_wind * shrink_factor
                
                # Apply weather adjustment to storm and fire perils in valid
                # The weather adjustment is a multiplier on the peril's cost component
                # For each row in valid, compute the weather multiplier for storm and fire
                hail_z_va = (hail_va - hail_mean_tr) / hail_std_tr
                wind_z_va = (wind_va - wind_mean_tr) / wind_std_tr
                
                weather_mult = 1.0 + alpha_hail_shrunk * hail_z_va + beta_wind_shrunk * wind_z_va
                weather_mult = np.maximum(weather_mult, 0.5)  # floor at 0.5 to avoid extreme values
                weather_mult = np.minimum(weather_mult, 2.0)  # cap at 2.0
                
                # Apply to storm and fire components
                # We need to decompose pred into peril components and multiply storm and fire by weather_mult
                # But pred is already adjusted by peril_factors. Let's recompute the valid_adjustment with weather
                # This is complex. Instead, let's apply the weather adjustment as a separate multiplier on the total pred
                # But that would apply it to all perils, not just storm and fire
                # Let's do it properly: recompute the valid_adjustment with weather
                valid_adjustment_with_weather = np.zeros(len(valid))
                for pf, ps in zip(peril_cols_f, peril_cols_s):
                    peril_name = pf.replace("prior_f_", "")
                    if peril_name not in peril_factors:
                        continue
                    
                    f_va = valid[pf].to_numpy(float)
                    s_va = valid[ps].to_numpy(float)
                    peril_cost_va = f_va * s_va
                    
                    denom_va = np.where(p_va > 0, p_va, 1.0)
                    w_va = peril_cost_va / denom_va
                    
                    # Apply weather multiplier to storm and fire
                    if peril_name in ["storm", "fire"]:
                        peril_factor_adj = peril_factors[peril_name] * weather_mult
                    else:
                        peril_factor_adj = peril_factors[peril_name]
                    
                    valid_adjustment_with_weather += w_va * peril_factor_adj
                
                pred = pred * valid_adjustment_with_weather

    return pred
