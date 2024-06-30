import torch
import tarfile
import torchaudio


def tarfile_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    # if there is more than one tar file in the dataset, then we need to
    # select one tar file for this worker
    print(f"Worker {worker_id} creating new tar file handles.")

    # reopen the tarfile handles for this worker
    dataset.tar_handles = []
    for tar_file in dataset.tar_files:
        tar_handle = tarfile.open(tar_file, "r:")
        tar_handle.next()  # skip the first root directory
        tar_handle.next()  # skip the first example directory
        dataset.tar_handles.append(tar_handle)


def torchaudio_decode(b):
    x, sr = torchaudio.load(b, format="FLAC")
    return x
