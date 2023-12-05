const tf = PyNULL()
const np = PyNULL()
const layer = PyNULL()
const euler_gpu = PyNULL()

function __init__()
    copy!(tf, pyimport("tensorflow"))
    copy!(np, pyimport("numpy"))
    copy!(layer, pyimport("deepreg.model.layer"))
    copy!(euler_gpu, pyimport("euler_gpu"))
end