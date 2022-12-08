



def flatten_dict(dd, prefix=''):
    return {k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}

