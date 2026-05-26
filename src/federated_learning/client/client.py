import torch

import flwr as fl
from collections import OrderedDict


from architectures.torch.implementation import train, evaluate

# Federated Learning Client
class FLClient(fl.client.NumPyClient):

    def __init__(self, 
                 *args,
                 cid=-1,
                 model=None,
                 i_epochs=5,
                 model_name="MOBILENET",
                 batch_size=32,
                 dataset="CIFAR-10",
                 strategy="fedavg",
                 model_path="",
                 result_path="",
                 computation_time_path="",
                 logger=None,
                 optimizer=None,
                 criterion=None,
                 scheduler=None,
                 device=None,
                 trainloader=None,
                 testloader=None,
                 **kwargs):
        
        # paths
        self.strategy = strategy
        self.dataset = dataset
        self.model_name = model_name
        self.model_path = model_path+model_name+'/'
        self.result_path = result_path+model_name+'/'
        self.time_path = computation_time_path+model_name+'/'
        self.logger = logger
        self.global_epoch = 0
        
        # identifiers
        self.cid = cid

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # client's data
        self.trainloader = trainloader 
        self.testloader = testloader 
        self.train_size = len(trainloader.dataset)
        self.test_size = len(testloader.dataset)

        # learning parameters
        self.i_epochs = i_epochs
        self.bs = batch_size

    def get_weights(self):
        
        result = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        return result
        
    def set_weights(self, 
                    parameters):
    
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # Implementar a função para o download de modelos
    def download_model(self):

        return 0

    # Implementar a função para o upload de modelos
    def upload_model(self):

        return 0

    # Implementar a função para o cálculo do atraso computacional
    def computational_time(self):

        return 0
    
    def get_properties(self, 
                       config):

        return {'cid': self.cid}

    def fit(self, 
            parameters, 
            config):

        communication_time = 0
        
        self.logger.debug("determine client's communication time to upload the model")
        communication_time += self.download_model()
        
        self.logger.debug("updating model parameters")
        self.set_weights(parameters)

        self.logger.debug("training model")
        loss = train(self.model, 
                     self.i_epochs, 
                     self.optimizer, 
                     self.criterion,
                     self.scheduler,
                     self.device,
                     self.trainloader,
                     self.logger)

        self.logger.debug("determine client's computational time")
        computational_time = computational_time()

        self.logger.debug("determine client's communication time to upload the model")
        communication_time += self.upload_model()

        total_training_time = computational_time + communication_time
        
        self.logger.debug(f'sending parameters to server: model_weights, len(train): {self.train_size}')
        return self.get_weights(), len(self.trainloader.dataset), {'loss':loss, "cid":self.cid, "training_time":total_training_time}

    def evaluate(self, 
                 parameters, 
                 config):
        
        self.logger.debug(f'evaluating model')  
        
        self.logger.debug("updating model parameters")
        self.set_weights(parameters)
 
        self.logger.debug("evaluating model")
        accuracy, loss = evaluate(self.model,
                                  self.device,
                                  self.criterion,
                                  self.testloader,
                                  self.logger)

        self.logger.debug(f'sending parameters to server: loss {loss}, len(test): {self.test_size} accuracy: {float(accuracy)}')
        return loss, self.test_size, {"accuracy": float(accuracy), "cid":self.cid}

