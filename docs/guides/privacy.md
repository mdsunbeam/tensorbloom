# Privacy and usage events

TensorBloom runs on your computer. Your models, code, datasets, files, and
training results stay there.

## Anonymous usage events (opt-in)

The first time you open TensorBloom, it asks whether you want to share
anonymous usage events. Nothing is sent unless you choose **Share**. You can
change your mind any time under **Help → About TensorBloom**.

These events tell us where people get stuck, for example how many new users
manage to finish their first training run. If you opt in, TensorBloom sends an
event when:

| Event | When | Extra details |
| --- | --- | --- |
| `app_opened` | TensorBloom starts | whether this is the first time it has run |
| `pytorch_installed` | PyTorch finishes installing | none |
| `training_finished` | A training run ends | whether it's your first, whether it stopped early, number of epochs, how long it took in seconds, and the device type (`cpu`, `cuda`, `mps`, or `gpu`) |
| `try_it_opened` | You open Try it on a run | whether it's your first time |

Every event also carries:

- a random ID made when TensorBloom first runs, so events from the same
  install can be counted together. It isn't tied to you, your computer, or
  any account.
- the TensorBloom version
- the operating system family: Windows, macOS, or Linux
- how many minutes have passed since TensorBloom first ran

Like any web request, the server that receives events sees your IP address.
It isn't kept with the events.

That is everything. Events never include your models, layer names, code, file
names or paths, datasets, training numbers such as loss or accuracy, GPU
names, or anything you type.

## Feedback

**Help → Send Feedback** opens a feedback form on GitHub with your TensorBloom
version and operating system filled in. You can edit or remove them before
posting. Posting needs a GitHub account, and the feedback is public.
