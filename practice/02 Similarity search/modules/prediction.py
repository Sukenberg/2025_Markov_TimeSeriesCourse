import numpy as np
import math

from modules.bestmatch import UCR_DTW, topK_match
import mass_ts as mts


default_match_alg_params = {
    'UCR-DTW': {
        'topK': 3,
        'r': 0.05,
        'excl_zone_frac': 1,
        'is_normalize': True
    },
    'MASS': {
        'topK': 3,
        'excl_zone_frac': 1
    }
}


class BestMatchPredictor:
    """
    Predictor based on best match algorithm
    """

    def __init__(self, h: int = 1, match_alg: str = 'UCR-DTW', match_alg_params: dict | None = None, aggr_func: str = 'average') -> None:
        """ 
        Constructor of class BestMatchPredictor

        Parameters
        ----------    
        h: prediction horizon
        match_algorithm: name of the best match algorithm
        match_algorithm_params: input parameters for the best match algorithm
        aggr_func: aggregate function
        """

        self.h: int = h
        self.match_alg: str = match_alg
        self.match_alg_params: dict | None = default_match_alg_params[match_alg].copy()
        if match_alg_params is not None:
            self.match_alg_params.update(match_alg_params)
        self.agg_func: str = aggr_func


    def _calculate_predict_values(self, topK_subs_predict_values: np.array) -> np.ndarray:
        """
        Calculate the future values of the time series using the aggregate function

        Parameters
        ----------
        topK_subs_predict_values: values of time series, which are located after topK subsequences

        Returns
        -------
        predict_values: prediction values
        """

        match self.agg_func:
            case 'average':
                predict_values = topK_subs_predict_values.mean(axis=0).round()
            case 'median':
                predict_values = topK_subs_predict_values.median(axis=0).round()
            case _:
                raise NotImplementedError
        
        return predict_values


    def predict_forecast(ts, query, h, match_alg, match_alg_params, aggr_func='average'):
        predict_values = np.zeros((h,))

        # Находим topK похожих подпоследовательностей
        if match_alg == 'MASS':
            # Используем MASS для поиска похожих подпоследовательностей
            distances = mts.mass(ts, query)
            # Получаем topK индексов из расстояний
            topK = match_alg_params['topK']
            best_indices = np.argsort(distances)[:topK]

        print(f"Найдены индексы {match_alg}: {best_indices}")

        # Собираем значения, которые следуют за найденными подпоследовательностями
        topK_subs_predict_values = []
        for idx in best_indices:
            # Проверяем, что после подпоследовательности есть достаточно значений для горизонта прогнозирования
            end_idx = idx + len(query)
            if end_idx + h <= len(ts):
                future_values = ts[end_idx:end_idx + h]
                topK_subs_predict_values.append(future_values)
                print(f"Индекс {idx}: будущие значения {future_values[:3]}... (среднее: {np.mean(future_values):.2f})")

        # Если нашли хотя бы одну подпоследовательность с будущими значениями
        if topK_subs_predict_values:
            topK_subs_predict_values = np.array(topK_subs_predict_values)
            print(f"Собрано {len(topK_subs_predict_values)} последовательностей")
            # Применяем агрегатную функцию для получения прогноза
            predict_values = topK_subs_predict_values.mean(axis=0)
        else:
            print("Не найдено подходящих подпоследовательностей с будущими значениями")

        return predict_values
