import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        #self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_dir = os.path.join(opt.switch_paths)
        #self.save_dir = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/Pix2Pix_and_District-classifier-modifica/checkpoints/Pix2Pix-Head_Pix2Pix-total'
        #self.save_dir_rebecca = '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Pix2Pix_WholeBody/checkpoints/experiment_name'
        self.save_dir_rebecca = '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Pix2Pix_Legs/checkpoints/experiment_name'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.to(device=gpu_ids[0])

    # helper loading function that can be used by subclasses

    # LOAD CHECKPOINT DEL MIO MODELLO
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        print(f"Successfully loaded {save_filename} from {save_path}")

    # LOAD CHECKPOINT MODELLO DI REBECCA
    def load_network1(self, network, network_label, epoch_label): # checkpoint Rebecca
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir_rebecca, save_filename)

        # Controlla se il file esiste prima di caricarlo
        if not os.path.isfile(save_path):
            raise FileNotFoundError(f"Checkpoint not found: {save_path}")

        # Carica il checkpoint
        checkpoint = torch.load(save_path)
        #print(f"Keys in the checkpoint {save_path}: {checkpoint.keys()}")

        # Ottieni lo stato attuale del modello
        model_state_dict = network.state_dict()
        checkpoint_keys = list(checkpoint.keys())  # Chiavi del checkpoint
        model_keys = list(model_state_dict.keys())  # Chiavi del modello

        #print("\n--- Dimensioni dei pesi nel checkpoint ---")
        # Stampa le dimensioni dei pesi nel checkpoint
        #for key in checkpoint_keys:
            #print(f"Checkpoint - {key}: {checkpoint[key].size()}")

        #print("\n--- Dimensioni dei pesi nel modello ---")
        # Stampa le dimensioni dei pesi nel modello
        #for key in model_keys:
            #print(f"Model - {key}: {model_state_dict[key].size()}")


        #if len(checkpoint_keys) < len(model_keys):
            #print("The checkpoint contains fewer keys than the model.")

        #print("\n--- Print checkpoint ---")
        #print(checkpoint.keys())

        #print("\n--- Print modello ---")
        #print(network)


        # Mappatura dinamica delle chiavi
        remapped_state_dict = {}
        for model_key, checkpoint_key in zip(model_keys, checkpoint_keys):
            print(f"Mapping: {checkpoint_key} -> {model_key}")
            remapped_state_dict[model_key] = checkpoint[checkpoint_key]

        # Aggiorna lo stato del modello con i valori mappati
        try:
            model_state_dict.update(remapped_state_dict)
            network.load_state_dict(model_state_dict, strict=False)
            print(f"Successfully loaded {network_label} from {save_path}")
            #torch.save(network.state_dict(), "temporary_model.pth")
        except Exception as e:
            print(f"Error loading state_dict for {network_label}: {e}")
            raise e









#-----------------------------------------------------

    def load_network4(self, network, network_label, epoch_label):
        import os
        import torch

        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir_rebecca, save_filename)

        # Controlla se il file esiste
        if not os.path.isfile(save_path):
            raise FileNotFoundError(f"Checkpoint not found: {save_path}")

        # Carica il checkpoint
        checkpoint = torch.load(save_path)
        print(f"Keys in the checkpoint {save_path}: {checkpoint.keys()}")

        # Ottieni lo stato attuale del modello
        model_state_dict = network.state_dict()
        print(f"Keys in the model: {model_state_dict.keys()}")

        # Mappatura esplicita delle chiavi
        key_mapping = {
            # Strato 1
            "conv1.weight": "model.model.0.weight",
            "conv1.bias": "model.model.0.bias",
            "conv2.weight": "model.model.1.model.1.weight",
            "conv2.bias": "model.model.1.model.1.bias",
            "conv3.weight": "model.model.1.model.3.model.1.weight",
            "conv3.bias": "model.model.1.model.3.model.1.bias",
            "conv4.weight": "model.model.1.model.3.model.3.model.1.weight",
            "conv4.bias": "model.model.1.model.3.model.3.model.1.bias",
            "conv5.weight": "model.model.1.model.3.model.3.model.3.model.1.weight",
            "conv5.bias": "model.model.1.model.3.model.3.model.3.model.1.bias",
            "convt1.weight": "model.model.1.model.3.model.3.model.3.model.3.weight",
            "convt1.bias": "model.model.1.model.3.model.3.model.3.model.3.bias",
            "convt2.weight": "model.model.1.model.3.model.3.model.5.weight",
            "convt2.bias": "model.model.1.model.3.model.3.model.5.bias",
            "convt3.weight": "model.model.1.model.3.model.5.weight",
            "convt3.bias": "model.model.1.model.3.model.5.bias",
            "convt4.weight": "model.model.1.model.5.weight",
            "convt4.bias": "model.model.1.model.5.bias",
            "convt5.weight": "model.model.3.weight",
            "convt5.bias": "model.model.3.bias",
        }

        # Crea un dizionario rimappato
        remapped_state_dict = {}
        for model_key, checkpoint_key in key_mapping.items():
            if checkpoint_key in checkpoint:
                remapped_state_dict[model_key] = checkpoint[checkpoint_key]
                print(f"Mapping: {checkpoint_key} -> {model_key}")
            else:
                print(f"Key {checkpoint_key} not found in checkpoint.")

        print("\n--- Print modello ---")
        print(network)

        # Aggiorna lo stato del modello con i valori mappati
        try:
            model_state_dict.update(remapped_state_dict)
            network.load_state_dict(model_state_dict, strict=False)
            print(f"Successfully loaded {network_label} from {save_path}")
        except Exception as e:
            print(f"Error loading state_dict for {network_label}: {e}")
            raise e

    def update_learning_rate():
        pass

