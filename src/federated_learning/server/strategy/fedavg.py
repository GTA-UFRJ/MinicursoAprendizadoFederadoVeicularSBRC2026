import time

import flwr as fl

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import Parameters, FitRes

from logging import WARNING


from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate_inplace, aggregate




class FedAvg(fl.server.strategy.FedAvg):

    def __init__(self,
                 *args,
                 logger=None,
                 time_path="",
                 **kwargs):
        
        super().__init__(*args,
                         **kwargs)

        self.logger = logger

        self.time_path = time_path
     
    def save_epoch_time(self,
                        epoch):

        with open(f"{self.time_path}training_time.csv", "a") as writer:
            
            line = f'{epoch},{self.global_agg_time}\n'

            writer.writelines(line)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        global_agg_start_time = time.time()

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.global_agg_time = time.time() - global_agg_start_time
        
        # save time measurement
        self.save_epoch_time(server_round)

        return parameters_aggregated, metrics_aggregated
