
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pix3d':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix3d_model import Pix2Pix3dModel
        model = Pix2Pix3dModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
