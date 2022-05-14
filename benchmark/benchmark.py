"""
Stage and run a Locust benchmark using strings from a test set.
"""
import os
import json
import pandas as pd
import click


@click.command()
@click.argument('dataset_name')
def main(dataset_name):
    """Extract string inputs from a dataset and use it to benchmark a server.

    Pass to it the name of an intermediate dataset to use strings from there.
    """
    path = '../datasets/{}/test.parquet'.format(dataset_name)

    test_set = pd.read_parquet(path)

    json_content = {
        'strings': test_set['name'].tolist()
    }

    with open('text-strings.json', 'w') as jsonfile:
        json.dump(json_content, jsonfile)

    os.system('locust --csv-full-history --csv=benchmark_{}'.format(dataset_name))

if __name__ == '__main__':
    main()
