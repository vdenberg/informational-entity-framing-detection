import argparse
from subprocess import Popen, PIPE


def run(args):
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        print('Stderr: ', args, '--->', stderr)
    return stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSC experiment', usage='singleEM [-noise] [-initialize] [-true_theta] ')
    parser.add_argument('-reproc', '--startiter', type=int)
    parser.add_argument('-e', '--enditer', type=int)
    parser.add_argument('-bit', '--bilty_train_iters', type=int)
    args = parser.parse_args()

    stit = args.startiter
    maxit = args.enditer
    for seed in [183, 212, 3456]:
        for fold in range(1,11):
            log_name = f"ssc_{seed}_{fold}"
            run(['scripts/train.sh',str(seed),str(fold),log_name])


