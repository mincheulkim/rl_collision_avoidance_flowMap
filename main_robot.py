import argparse



def main():
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs')
    args = parser.parse_args()

    configs = []
    main(configs)