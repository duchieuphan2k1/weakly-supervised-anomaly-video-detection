class Config(object):
    def __init__(self, args):
        self.lr = eval("[{}]*{}".format(args.lr, args.max_epoch))
        self.lr_str = "[{}]*{}".format(args.lr, args.max_epoch)

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')