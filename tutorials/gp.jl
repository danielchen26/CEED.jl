using DataFrames, Random

# Set a random seed for reproducibility
Random.seed!(123)

# Generate simulated data
n = 1000  # number of data points
data = DataFrame()
for i in 1:5
    data[!, "feature_$i"] = rand(n)
    data[!, "uncertainty_$i"] = rand(n)
end

data


using Distances, GaussianProcesses

# Define a distance measure
distance = Euclidean()

# Define a kernel function based on the distance measure
# kernel = transform(SEIso(1.0, 1.0), distance)
kernel = SEIso(1.0, 1.0)



# Define the features and uncertainties
features = ["feature_$i" for i = 1:5]
uncertainties = ["uncertainty_$i" for i = 1:5]

# Fit a Multivariate Gaussian Process using all features
gp = GP(Matrix(data[!, features]), Matrix(data[!, uncertainties]), MeanZero(), kernel)


# Fit a separate Gaussian Process for each output variable
gps = [
    GP(data[!, feature], data[!, uncertainty], MeanZero(), kernel) for
    (feature, uncertainty) in zip(features, uncertainties)
]

# Define a new sampler that samples from each Gaussian Process
sampler = function (evidence::Evidence, columns, rng = default_rng())
    # Sample from each Gaussian Process
    observed = [rand(rng, gp) for gp in gps]
    return Dict(c => observed[i] for (i, c) in enumerate(columns))
end

# Update each Gaussian Process with new data
gps = [
    GP(data[!, feature], data[!, uncertainty], MeanZero(), kernel) for
    (feature, uncertainty) in zip(features, uncertainties)
]