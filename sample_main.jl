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
        ("--batchsize"; arg_type=Int; default=1000; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--output"; arg_type=String; default="./"; help="output dir")
        ("--fp64"; action=:store_true; help="use `Float64` (default: `Float32`)")
    end
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    parsed_args = parse_args(args, s; as_symbols=true)
    batchsize = parsed_args[:batchsize]
    epochs = parsed_args[:epochs]
    outputdir = parsed_args[:output]
    FType = parsed_args[:fp64] ? Float64 : Float32

    model = FluxMNIST.Model{FType}()
    traindata, tX, tY = FluxMNIST.loadMNIST(FType, batchsize)
    accuracy = FluxMNIST.Accuracy(model.m)
    evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
    @time FluxMNIST.train!(model, traindata; epochs=epochs, cb=evalcb)
    @show(accuracy(tX, tY))
    # FluxMNIST.savemodel(model)
    savemodel(model, outputdir)
end

abspath(PROGRAM_FILE) == @__FILE__() && main(ARGS)