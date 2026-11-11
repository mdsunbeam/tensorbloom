/** Choices the Training panel offers, shared with the assistant so it can only pick what the panel can */
export const OPTIMIZERS = ["SGD", "Adam", "AdamW", "RMSprop", "Adadelta", "Adagrad"];
export const SCHEDULERS = ["None", "StepLR", "CosineAnnealingLR", "OneCycleLR", "ReduceLROnPlateau", "ExponentialLR"];
export const LOSSES = ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss", "NLLLoss", "HuberLoss"];
// Every device training accepts; the Training panel offers the ones that fit this computer (see devices.ts)
export const DEVICES = ["auto", "cpu", "mps", "cuda:0", "cuda:1"];
