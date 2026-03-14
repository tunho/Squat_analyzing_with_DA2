from __future__ import annotations

from typing import Any

try:
    from squat.analyzer.base import BaseSquatAnalyzer
    from squat.calibration.relative_models import (
        compute_model_angles,
        fit_model_from_history,
        get_relative_model,
    )
except ImportError:
    from analyzer.base import BaseSquatAnalyzer
    from calibration.relative_models import (
        compute_model_angles,
        fit_model_from_history,
        get_relative_model,
    )


class RelativeModelSquatAnalyzer(BaseSquatAnalyzer):
    def __init__(self, model: str = "relative_linear", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = get_relative_model(model)

    def finalize_analysis(self):
        if not self.history_data:
            return 0, [], None

        params = fit_model_from_history(self.history_data, self.model)
        if params is None:
            return self.counter.count, [], None

        final_angles = compute_model_angles(self.history_data, self.model, params)
        return self.counter.count, final_angles, params