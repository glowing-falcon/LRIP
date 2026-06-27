import schedulefree


def scheduler_free_optimizer(optimizer_name, params, *args, **kwargs):
    return getattr(schedulefree, optimizer_name)(params, *args, **kwargs)
