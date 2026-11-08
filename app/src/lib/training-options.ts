/** Choices the Training panel offers, shared with the assistant so it can only pick what the panel can */
export const OPTIMIZERS = ["SGD", "Adam", "AdamW", "RMSprop", "Adadelta", "Adagrad"];
export const SCHEDULERS = ["None", "StepLR", "CosineAnnealingLR", "OneCycleLR", "ReduceLROnPlateau", "ExponentialLR"];
export const LOSSES = ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss", "NLLLoss", "HuberLoss"];
export const DEVICES = ["auto", "cpu", "cuda:0", "cuda:1"];
