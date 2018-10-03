using Random
using Base.Iterators: partition

_slice(x::AbstractVector, idxs) = x[idxs]
_slice(x::AbstractMatrix, idxs) = x[:, idxs]
@generated function _slice(x::AbstractArray{T,N}, idxs) where {T,N}
    colons = [(:) for _ in 1:N-1]
    :(x[$(colons...), idxs])
end

struct BatchProducer{A1<:AbstractArray,A2<:AbstractArray}
    x::A1
    y::A2
    batchsize::Int
    shuffle::Bool
end
Base.IteratorSize(::Type{BatchProducer}) = Base.HasLength()
Base.length(t::BatchProducer) = cld(size(t.x)[end], t.batchsize)
Base.IteratorEltype(::Type{BatchProducer}) = Base.HasEltype
Base.eltype(::Type{BatchProducer{A1,A2}}) where {A1,A2} = Tuple{A1,A2}

function Base.iterate(t::BatchProducer)
    l = size(t.x)[end]
    idxss = if t.shuffle
        [idxs for idxs in partition(shuffle(1:l), t.batchsize)]
    else
        [idxs for idxs in partition(1:l, t.batchsize)]
    end
    iterate(t, (idxss, iterate(idxss)))
end
Base.iterate(t::BatchProducer, (idxss, _)::Tuple{Vector{<:AbstractVector{Int}},Nothing}) = nothing
function Base.iterate(t::BatchProducer, (idxss, (idxs, st))::Tuple{Vector{<:AbstractVector{Int}},Tuple{AbstractVector{Int},Any}})
    ((_slice(t.x, idxs), _slice(t.y, idxs)), (idxss, iterate(idxss, st)))
end
