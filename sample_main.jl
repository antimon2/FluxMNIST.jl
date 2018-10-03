using FluxMNIST
using Flux: throttle
using ArgParse
using Dates

function savemodel(model::FluxMNIST.Model{T}, outputdir::AbstractString) where {T<:AbstractFloat}
    if !isdir(outputdir)
        mkpath(outputdir)
    end
    ts = Dates.format(now(), dateformat"yyyymmddHHMMSS")
    filename = "model-flux-f$(sizeof(T)*8)_$(ts).bson"
    dst = joinpath(outputdir, filename)
    FluxMNIST.savemodel(model, dst)
end

main(args::AbstractString) = main(split(args))
function main(args::Vector{<:AbstractString}=[])
    s = ArgParseSettings()
    s.description="FluxMNIST.jl sample for Float32"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        # ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=1000; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--output"; arg_type=String; default="./"; help="output dir")
        # ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        # ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        # ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        # ("--fast"; action=:store_true; help="skip loss printing for faster run")
        # ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        # ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)
    batchsize = o[:batchsize]
    epochs = o[:epochs]
    outputdir = o[:output]

    model = FluxMNIST.Model{Float32}()
    traindata, tX, tY = FluxMNIST.loadMNIST(Float32, batchsize)
    accuracy = FluxMNIST.Accuracy(model.m)
    evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
    @time FluxMNIST.train!(model, traindata; epochs=epochs, cb=evalcb)
    @show(accuracy(tX, tY))
    # FluxMNIST.savemodel(model)
    savemodel(model, outputdir)
end

abspath(PROGRAM_FILE) == @__FILE__() && main(ARGS)