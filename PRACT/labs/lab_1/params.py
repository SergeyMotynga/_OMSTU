PARAM_SPACES = {
    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive": [True, False],
    },

    "RandomForestRegressor": {
        "n_estimators": ("int", 50, 300, {"step": 25}),
        "max_depth": ("int", 3, 30, {"step": 1}),
        "min_samples_split": ("int", 2, 20, {"step": 1}),
        "min_samples_leaf": ("int", 1, 20, {"step": 1}),
    },

    "GradientBoostingRegressor": {
        "n_estimators": ("int", 50, 600, {"step": 50}),
        "learning_rate": ("float", 1e-3, 0.3, {"log": True}),
        "max_depth": ("int", 2, 10, {"step": 1}),
        "min_samples_split": ("int", 2, 20, {"step": 1}),
        "min_samples_leaf": ("int", 1, 20, {"step": 1}),
        "subsample": ("float", 0.5, 1.0),
    },

    "XGBRegressor": {
        "n_estimators": ("int", 100, 800, {"step": 50}),
        "learning_rate": ("float", 1e-3, 0.3, {"log": True}),
        "max_depth": ("int", 3, 12, {"step": 1}),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("float", 1e-8, 1.0, {"log": True}),
        "reg_lambda": ("float", 1e-8, 10.0, {"log": True}),
        "min_child_weight": ("float", 1e-3, 10.0, {"log": True}),
    },

    "LGBMRegressor": {
        "n_estimators": ("int", 100, 1000, {"step": 50}),
        "learning_rate": ("float", 1e-3, 0.3, {"log": True}),
        "max_depth": ("int", -1, 16, {"step": 1}),  # -1 = no limit
        "num_leaves": ("int", 15, 255, {"step": 5}),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_samples": ("int", 5, 100, {"step": 5}),
        "reg_alpha": ("float", 1e-8, 1.0, {"log": True}),
        "reg_lambda": ("float", 1e-8, 10.0, {"log": True}),
    },

    "CatBoostRegressor": {
        "iterations": ("int", 300, 1500, {"step": 100}),
        "depth": ("int", 4, 10, {"step": 1}),
        "learning_rate": ("float", 1e-3, 0.3, {"log": True}),
        "l2_leaf_reg": ("float", 1.0, 10.0, {"log": True}),
        "bagging_temperature": ("float", 0.0, 1.0),
        "loss_function": ["RMSE", "MAE"],
    },
}
