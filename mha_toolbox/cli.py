import argparse
import os
import json
from mha_toolbox.toolbox import MHAToolbox


def _load_dataset(name):
    if name is None:
        return None, None
    name = name.lower()
    if name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target
    raise ValueError(f"Unknown dataset '{name}'")


def main():
    parser = argparse.ArgumentParser(prog='mha', description='MHA Toolbox CLI')
    sub = parser.add_subparsers(dest='command')

    # list
    list_p = sub.add_parser('list', help='List available algorithms')

    # info
    info_p = sub.add_parser('info', help='Show info for an algorithm')
    info_p.add_argument('algorithm')

    # run
    run_p = sub.add_parser('run', help='Run an algorithm')
    run_p.add_argument('algorithm')
    run_p.add_argument('--dataset', default=None, help='Dataset name (e.g. breast_cancer)')
    run_p.add_argument('--population_size', type=int, default=None)
    run_p.add_argument('--max_iterations', type=int, default=None)
    run_p.add_argument('--dimensions', type=int, default=None)
    run_p.add_argument('--output', default='results', help='Output folder')
    run_p.add_argument('--save_model', action='store_true', help='Also pickle the model')

    args = parser.parse_args()
    toolbox = MHAToolbox()

    if args.command == 'list':
        names = toolbox.get_all_algorithm_names()
        for n in names:
            print(n)
        return

    if args.command == 'info':
        info = toolbox.get_algorithm_info(args.algorithm)
        print(json.dumps(info, indent=2, default=str))
        return

    if args.command == 'run':
        X, y = _load_dataset(args.dataset)
        params = {}
        if args.population_size is not None:
            params['population_size'] = args.population_size
        if args.max_iterations is not None:
            params['max_iterations'] = args.max_iterations
        if args.dimensions is not None:
            params['dimensions'] = args.dimensions

        print(f"Running {args.algorithm} with params: {params} dataset={args.dataset}")
        result = toolbox.optimize(args.algorithm, X=X, y=y, **params)

        # ensure output dir exists
        outdir = args.output
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, f"{args.algorithm}_{args.dataset or 'run'}")
        primary = result.save(base + '.json')
        if args.save_model:
            result.save_model(base + '_model.pkl')

        print(f"Saved results to {primary}")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
