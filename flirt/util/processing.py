import tempfile
import os

import time
from joblib import load, dump, _memmapping_reducer

# Some system have a ramdisk mounted by default, we can use it instead of /tmp
# as the default folder to dump big arrays to share with subprocesses.
SYSTEM_SHARED_MEM_FS = '/dev/shm'

# Minimal number of bytes available on SYSTEM_SHARED_MEM_FS to consider using
# it as the default folder to dump big arrays to share with subprocesses.
SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(2e9)


def memmap_auto(data, callable, **args):
    """
    Takes care of memmapping data for optimal processing
    :param data: the data to be memmapped, deleted after memmapping
    :param callable: the callable to be ecexuted, first argument is memmapped data
    :param args: optional args to be supplied to callable
    :return: the return value from callable
    """
    memmapped_data, memmap_filename = memmap_data(data)
    del data
    result = callable(memmapped_data, **args)
    memmap_unlink(memmap_filename)
    return result


def memmap_data(data, read_only: bool = True):
    new_folder_name = ("flirt_memmap_%d" % os.getpid())
    temp_dir, _ = __get_temp_dir(new_folder_name)
    filename = os.path.join(temp_dir, 'memmap_%s.mmap' % next(tempfile._RandomNameSequence()))
    if os.path.exists(filename):
        os.unlink(filename)
    _ = dump(data, filename)
    return load(filename, mmap_mode='r+' if read_only else 'w+'), filename


# taken from https://github.com/joblib/joblib/blob/c952c223e3616e5ff10f63864b4615088b0d9352/joblib/_memmapping_reducer.py
def memmap_unlink(filename):
    """Wrapper around os.unlink with a retry mechanism.

    The retry mechanism has been implemented primarily to overcome a race
    condition happening during the finalizer of a np.memmap: when a process
    holding the last reference to a mmap-backed np.memmap/np.array is about to
    delete this array (and close the reference), it sends a maybe_unlink
    request to the resource_tracker. This request can be processed faster than
    it takes for the last reference of the memmap to be closed, yielding (on
    Windows) a PermissionError in the resource_tracker loop.
    """
    if os.path.exists(filename):
        NUM_RETRIES = 10
        for retry_no in range(1, NUM_RETRIES + 1):
            try:
                os.unlink(filename)
                break
            except PermissionError:
                if retry_no == NUM_RETRIES:
                    print("Unable to remove memmapped file")
                    break
                else:
                    time.sleep(.2)


def __get_temp_dir(pool_folder_name, temp_folder=None):
    use_shared_mem = False
    if temp_folder is None:
        if os.path.exists(SYSTEM_SHARED_MEM_FS):
            try:
                shm_stats = os.statvfs(SYSTEM_SHARED_MEM_FS)
                available_nbytes = shm_stats.f_bsize * shm_stats.f_bavail
                if available_nbytes > SYSTEM_SHARED_MEM_FS_MIN_SIZE:
                    # Try to see if we have write access to the shared mem
                    # folder only if it is reasonably large (that is 2GB or
                    # more).
                    temp_folder = SYSTEM_SHARED_MEM_FS
                    pool_folder = os.path.join(temp_folder, pool_folder_name)
                    if not os.path.exists(pool_folder):
                        os.makedirs(pool_folder)
                    use_shared_mem = True
            except (IOError, OSError):
                # Missing rights in the /dev/shm partition, fallback to regular
                # temp folder.
                temp_folder = None
    if temp_folder is None:
        # Fallback to the default tmp folder, typically /tmp
        temp_folder = tempfile.gettempdir()
    temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
    pool_folder = os.path.join(temp_folder, pool_folder_name)
    if not os.path.exists(pool_folder):
        os.makedirs(pool_folder)
    return pool_folder, use_shared_mem
