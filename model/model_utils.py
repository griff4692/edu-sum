def infer_hf_model(args, is_abstract=False):
    if args.hf_model is None:
        # Infer it
        if args.dataset == 'samsum':
            args.hf_model = 'lidiya/bart-large-xsum-samsum'
        elif args.dataset == 'cnn_dailymail':
            args.hf_model = 'facebook/bart-large-cnn'
        elif args.dataset == 'nyt':
            args.hf_model = 'facebook/bart-large'
        elif args.dataset == 'xsum':
            if is_abstract:
                args.hf_model = 'google/pegasus-xsum'
            else:
                args.hf_model = 'facebook/bart-large-xsum'
        else:
            raise Exception('Unknown Yet.')
