import glob
import re
import sys
import subprocess
import tempfile
import time
import os

import shutil


def find_cupr_dir(dir):
    for (dirpath, _, _) in os.walk(dir):
        if re.search("cupr-\d+$", dirpath):
            return dirpath
    raise Exception("CUPR directory not found in {}".format(dir))


# https://stackoverflow.com/a/1094933/1107768
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python benchmark.py <path-to-program>")
        exit(1)

    program = os.path.abspath(sys.argv[1])
    buffer_size = 1024 * 1024 * 5
    formats = ["JSON", "PROTOBUF", "CAPNP"]
    compression = ["0", "1"]

    for format in formats:
        for comp in compression:
            args = [program]
            env = {
                "BUFFER_SIZE": str(buffer_size),
                "FORMAT": format,
                "COMPRESS": comp
            }

            dir = tempfile.mkdtemp("cupr-benchmark")

            start = time.time()

            process = subprocess.Popen(args,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       env=env,
                                       cwd=dir)
            (out, err) = process.communicate()
            assert process.returncode == 0

            duration = time.time() - start

            cuprdir = find_cupr_dir(dir)
            sizes = [os.stat(f).st_size for f in glob.glob("{}/*.trace.*".format(cuprdir))]
            file_size = sum(sizes) / len(sizes)

            print(out.strip())
            print("{}{}: duration {} s, file size {}\n".format(format, "" if comp == "0" else "/gzip",
                                                               duration, sizeof_fmt(file_size)))

            shutil.rmtree(dir, ignore_errors=False)
