### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ e5dd0431-45b4-46c5-8b42-24df6b00f4c0
using Flux: onehotbatch, onecold, crossentropy, throttle, params, argmax

# ╔═╡ cef296fc-8f9f-411d-882d-854bc6982146
using Flux: Data.DataLoader, Descent, ADAM

# ╔═╡ 66f4116d-8c85-4a17-a4c6-8043ffe7d7cf
using Flux: @epochs

# ╔═╡ 7421e5e6-d140-491f-b5ac-324a8c0a7661
using Flux, MLBase, GLMNet, Plots, Distances, DataStructures,  RDatasets,  MultivariateStats,  Lathe, DataFrames, BSON,  Images,  Printf, Statistics, Features

# ╔═╡ d9ef963a-cb58-4ae2-9cb0-3063f74732b9
using Flux: Conv, Dense, Chain, relu, softmax, CUDA

# ╔═╡ 6ed42c7a-3419-4939-9d8b-60d0ce3fa42a
using Base.Iterators: repeated, partition

# ╔═╡ 33cbeeb1-3dde-4238-a6bb-ad0ccfc31f29
using Lathe.preprocess: TrainTestSplit

# ╔═╡ dfcfb664-d1b2-404c-99fd-58733a913b89
test_data = readdir("C://Users//Mr OHMS//Desktop//2021 semester 1//Artificial Intelligence//Assignment02//chest_xray//test")

# ╔═╡ 87f1f15d-0571-4763-a120-0ec65ebc6cd9
test_normal = readdir("C://Users//Mr OHMS//Desktop//2021 semester 1//Artificial Intelligence//Assignment02//chest_xray//test//NORMAL")

# ╔═╡ 7a13316d-1a6f-4b0e-aead-55cb0df7b6e5
test_pneumonia = readdir("C://Users//Mr OHMS//Desktop//2021 semester 1//Artificial Intelligence//Assignment02//chest_xray//test//PNEUMONIA")

# ╔═╡ fb740e29-e0fc-4df1-9f6e-b0f188620bec
train_data = readdir("C://Users//Mr OHMS//Desktop//2021 semester 1//Artificial Intelligence//Assignment02//chest_xray//train")

# ╔═╡ f41ad983-0d1f-44d9-8204-02ed5571a237
train_normal = readdir("C://Users//Mr OHMS//Desktop//2021 semester 1//Artificial Intelligence//Assignment02//chest_xray//train//NORMAL")

# ╔═╡ cd29aa06-1eee-4439-8b14-a63f979dcddb
train_pneumonia = readdir("C://Users//Mr OHMS//Desktop//2021 semester 1//Artificial Intelligence//Assignment02//chest_xray//train//PNEUMONIA")

# ╔═╡ 4d29ac13-9012-4049-bd51-5ebe7dc23c83
size(test_normal)

# ╔═╡ dfc4a3ab-83fa-44b1-bf71-5ede63fcf39a
size(test_pneumonia)

# ╔═╡ 5666d982-1e2b-41e3-96a1-623c0f2beff0
size(train_normal)

# ╔═╡ a36bf38e-da11-4fa9-b7df-c5c2d62df844
size(train_pneumonia)

# ╔═╡ a6e51c25-fe10-446e-9506-acafb5dba8a7
train_normal[40]

# ╔═╡ 7c7b3c83-c227-4c83-909b-ce52229bac56
train_pneumonia[41]

# ╔═╡ a811db25-52c1-47f9-ad13-17bee1664b8b
test_normal[152]

# ╔═╡ 1b6e40f5-e7aa-4985-9c15-85c87316d836
train_pneumonia[214]

# ╔═╡ 27e9d7fa-f8ab-40ce-9a0c-a65c799fa741
typeof(train_normal[3])

# ╔═╡ e624cc4e-cf30-469b-9016-198e87add9c5
trainNormal = DataFrame(:Feature => train_normal, :Target => train_normal, :Binary => 0, :Bool => false)

# ╔═╡ 7f5b9f27-5753-4d5e-aa2e-bf4d4f7cee28
testNormal = DataFrame(:Feature => test_normal, :Target => test_normal, :Binary => 0, :Bool => false)

# ╔═╡ cfa3bbf4-fddb-404d-9a8a-03b05bc90f32
trainPneumonia = DataFrame(:Input => test_pneumonia, :Variables => test_pneumonia, :Binary => 1, :Bool => true)

# ╔═╡ 015c165c-5b5e-40cd-b286-99e2e7c16892
testPneumonia = DataFrame(:Input => test_pneumonia, :Variables => test_pneumonia, :Binary => 1, :Bool => true)

# ╔═╡ 68e13bfb-b17e-438d-8f6c-53e99a2926ce
train, val = TrainTestSplit(trainNormal[:,2], 0.2)

# ╔═╡ 911ffd22-c2fa-455d-98c6-46455ddd21bc
train2, val2 = TrainTestSplit(trainPneumonia[:,2], 0.2)

# ╔═╡ 2ec6642e-3a3f-42a2-abdb-bf2f5bf357d7
feat = :Feature

# ╔═╡ efd60a88-57a5-42c2-b9ed-a53af6c39bf8
targ = :Target

# ╔═╡ 695c5380-f5e4-4261-991e-60d91d9d644f
input = :Input

# ╔═╡ 1455bf85-05ec-4c16-9bac-e1f30b20c810
var = :Variables

# ╔═╡ 45448190-523f-4ac3-8e6f-5f81de52c4fb

X_normal = Matrix(trainNormal[:,:])

# ╔═╡ d4801ed9-ca9a-48d5-ba6a-37897dd3659b
X_test = Matrix(testPneumonia[:,:])

# ╔═╡ 1e7e4e24-6a0d-439c-8e3a-36fea1bd50c3
X_pneumonia = Matrix(trainPneumonia[:,:])

# ╔═╡ 631b6525-6de7-4e4b-be3a-e640268c3b38
X_testpneumomia = Matrix(testPneumonia[:,:])

# ╔═╡ 6470bd9c-9626-458b-9c4c-df0c058bbaa1
data = []

# ╔═╡ b32d08fd-af76-45ce-b3cd-47602dd4a8c6
for i in train_normal
	if (i[1] == ".jpeg")
		append!(data, "1")
	else
		append!(data, "0")
	end
end

# ╔═╡ a206d691-57e3-4528-ab0b-7f0579eb2342
data

# ╔═╡ f41dddef-b173-494c-896e-fc5c591503da
data_normal = []

# ╔═╡ ea1130c0-a9c4-4bf8-9c4d-c0c4af22ddca
for i in train_normal
	append!(data_normal, 0)
end

# ╔═╡ 25dc1648-d99a-403e-bfbb-ed7b7cd18ecc
data_normal

# ╔═╡ e50ab593-0524-4760-aff2-a45ddf094fdf
size(data_normal)

# ╔═╡ 8e04eb33-d25c-45f7-ab14-d484a3a5353c
data_pneumonia = []

# ╔═╡ 2ae6cf01-1690-4236-960a-e5e9606a3b4d
for i in train_pneumonia
	append!(data_pneumonia, 1)
end

# ╔═╡ f7cc9fb5-979d-4c57-be94-9aa264385591
data_pneumonia

# ╔═╡ 86c1acc8-81f8-47a7-953c-48f72fec8482
size(data_pneumonia)

# ╔═╡ 1a9ae560-8c95-478c-bc46-0fa49c0eef9a
data_set = []

# ╔═╡ 9f72674f-0907-4aec-8861-3cab74479859
for i in data_normal
	append!(data_set, data_normal)
end

# ╔═╡ 9eb0d907-5225-4b10-aa0b-d50a8ebf4792
for i in data_pneumonia
	append!(data_set, data_pneumonia)
end

# ╔═╡ 06be2f09-0a49-487a-be70-e79ec61554d6
data_set

# ╔═╡ 6e3f2c68-70f4-466d-b2da-6a5506216799
size(data_set)

# ╔═╡ fd930a19-4355-4cad-aefa-73cb1c307517
typeof(data_set)

# ╔═╡ d319751c-9eb7-4cff-8137-bc07bb2f934c
trained_normal = DataFrame(:Feature => train_normal, :Target => data_normal)

# ╔═╡ 13699545-2e1c-4eb1-bbe4-94cb9e770f7c
trained_pneumonia = DataFrame(:Input =>train_pneumonia, :Variabe => data_pneumonia)

# ╔═╡ 20e5febc-01ed-45cd-bb46-994324ee7262
training_1, varieble_1 = TrainTestSplit(trainNormal[:,2], 0.8)

# ╔═╡ 95cce0a0-e2d0-4b81-939d-741b34a39487
training_2, varieble_2 = TrainTestSplit(trainPneumonia[:,2], 0.8)

# ╔═╡ 8031ff2d-3430-4797-9146-1d9b87238fdd
normal_labels = X_normal[:,:]

# ╔═╡ 39f6a4e4-58e0-4373-a915-1d78ad6b3b35
normaltest_labels = X_test[:,:]

# ╔═╡ a6862cea-1cb6-4bc9-9b05-e74615b2cd3b
pneumonia_labels = X_pneumonia[:,:]

# ╔═╡ 9e79aa28-2987-4b89-9c69-d378703d65b5
pneumoniatest_labels = X_pneumonia[:,:]

# ╔═╡ 1a70f812-5b5a-4ba8-a966-50cfb957b4b7
normal_labels_map = labelmap(normal_labels)

# ╔═╡ 18cd770a-935f-44eb-9a77-3295442527b2
Xtest_labels_map = labelmap(normaltest_labels)

# ╔═╡ 1e669ffb-eba3-49c9-a727-616afbc8c8ca
pneumonia_labels_map = labelmap(pneumonia_labels)

# ╔═╡ b7ba5c7c-382f-4802-a3fa-d53e8d672979
pneumonia_test_label_map = labelmap(pneumoniatest_labels)

# ╔═╡ dede71e8-dca7-4380-8681-927092dd58c8
y_normal = labelencode(normal_labels_map, normal_labels)

# ╔═╡ 5aeee1f4-3712-425a-863d-75cdbecee847
y_normal_test = labelencode(Xtest_labels_map, normaltest_labels)

# ╔═╡ b3164ca1-fe99-4b32-8b8f-de16fdcaac33
y_pneumonia = labelencode(pneumonia_labels_map, pneumonia_labels)

# ╔═╡ 578675d7-136f-4474-ad57-2664e89cd194
y_pneumoniatest = labelencode(pneumonia_test_label_map, pneumoniatest_labels)

# ╔═╡ 50bec8e1-e1b1-4903-95b0-b9b87427bb78
training_splitNormal = TrainTestSplit(y_normal[:], 0.8 )

# ╔═╡ 76f81a2a-8d84-439c-8694-db6b63daf77f
test_ids = setdiff(1:length(y_normal))

# ╔═╡ a4fb1b74-3647-4712-a85d-1408be03a87c
training_splitPneumonia = TrainTestSplit(y_pneumonia[:], 0.8 )

# ╔═╡ 3b015457-bfb6-48d7-9d6d-aa6597b48175
test_idx = setdiff(1:length(y_pneumonia))

# ╔═╡ 9de323c0-e50e-404e-ba43-60ab6a1ddcea
assign_class(predictedvalue) = argmin(abs.(predictedvalue .- [1,2,3]))

# ╔═╡ c140cddc-e416-4105-90d3-1a80c7db11f8
q_normal = X_normal[test_ids]

# ╔═╡ c52730da-bfaa-43fe-bc47-fea5349995e6
q_pneumonia  = X_pneumonia[test_idx]

# ╔═╡ 9557f1c9-7d47-4e85-aa34-9de4489c6fe3
normal_vector = Vector{Int64}(undef, size(y_normal)[1])

# ╔═╡ e7be8846-d143-457c-880f-06c9846cf0aa
testnormal_vector = Vector{Int64}(undef, size(y_normal_test)[1])

# ╔═╡ c1e8b6b6-09ea-4e6f-86ef-78164e31bad5
typeof(normal_vector)

# ╔═╡ 239fa5df-bd65-49de-8739-7cc90b737983
pneumonia_vector = Vector{Int64}(undef, size(y_pneumonia)[1])

# ╔═╡ 925f4d88-6099-40ba-99f0-000f82a7961a
testpneumonia_vector = Vector{Int64}(undef, size(y_pneumoniatest)[1])

# ╔═╡ 6ee37de2-09cc-44c9-af1f-0c2c6e395823
train_normals = DataLoader(normal_vector, batchsize = 128)

# ╔═╡ 8b530b2a-9599-485d-9c9d-d0091edc70bb
train_pneumonias = DataLoader(pneumonia_vector, batchsize = 128)

# ╔═╡ a18424f5-b559-4d93-8f3c-2bf08dd23b7c
model = Chain(
Conv((3, 3), 1=>16, pad=(1,1), stride=2, relu),
    x -> maxpool(x, (2,2)),
Conv((3, 3), 16=>32, pad=(1,1), stride=2, relu),
    x -> maxpool(x, (2,2)),
Conv((3, 3), 32=>32, pad=(1,1), stride=2, relu),
Conv((3, 3), 32=>64, pad=(1,1), stride=2, relu),
    x -> reshape(x, :, size(x, 4)),
Dense(288, 10), softmax)

# ╔═╡ 32e21624-33b8-43c5-b036-ac3611817af2
lr = 0.1

# ╔═╡ 3aba7316-e3a0-44f7-867d-0f720d2ad366
optimizer = Descent(lr)

# ╔═╡ b8a6f4a3-2aa1-4291-a215-6108943aff88
number_epochs = 10

# ╔═╡ eed53fba-9e06-4204-a67b-88ac47cb61c3
typeof(train_normals)

# ╔═╡ 73612f7c-26f7-41b2-b390-f054d8af656d
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:,i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

# ╔═╡ db362bbf-9fa8-4b21-a9c5-640c2160e955
batch_size = 128

# ╔═╡ 37f6feaa-ca59-4f64-aabc-b3aa5467aec3
floatingpoint_image = float.(test_ids)

# ╔═╡ f41b32c2-3ae5-47fb-91c6-c612eab11822
X = hcat(train_normal...)

# ╔═╡ 9052c831-b80b-4222-8ae4-a4c424629047
X2 = hcat(normal_vector...)

# ╔═╡ 427ccc3f-d926-4b68-9e2c-36ae27a01788
optimz = ADAM()

# ╔═╡ 5f98aff3-8285-4342-9faf-e520ae995455
evalcb() = @show(loss(normal_vector, train_normals))

# ╔═╡ ccd496fd-c654-4dd9-80fe-70ae8b69d7f6
datasetx = repeated((y_normal, normal_vector),200)

# ╔═╡ aac22a22-b4d6-4148-ab6c-eab933ade2bd
function get_scaling_params(init_feature_mat)
feat_mean = mean(init_feature_mat, dims=1)
feat_dev = std(init_feature_mat, dims=1)
return (feat_mean, feat_dev)
end

# ╔═╡ d850ea8e-6a21-407e-a30b-a5f5da9d0393
scaling_params = get_scaling_params(y_normal)

# ╔═╡ a08cbbcd-19c7-4040-85c7-7bb1fedbea07
scaling_params2 = get_scaling_params(y_pneumonia)

# ╔═╡ 8e4cedcb-a3cd-49b8-a10e-bb61b13cb661
function scale_features(feature_mat, sc_params)
scaled_feature_mat = (feature_mat .- sc_params[1]) ./ sc_params[2]
end

# ╔═╡ ac2a0b34-eed2-4719-90df-7b3eadc75e0c
scaled_training_features = scale_features(y_normal, scaling_params)

# ╔═╡ a815ab6a-6aa2-4691-b77a-e75997ed1696
scaled_testing_features = scale_features(y_normal_test, scaling_params)

# ╔═╡ a73a2b9a-23b7-4bef-b3c9-88e5fab0ff5a
function get_cost(aug_features, outcome, weights, reg_param)
sample_count = length(outcome)
hypothesis = sigmoid(aug_features * weights)
cost_part_1 = ((-outcome)' * log.(hypothesis))[1]
cost_part_2 = ((1 .- outcome)' * log.(1 .- hypothesis))[1]
lambda_regul = (reg_param/(2 * sample_count) * sum(weights[2:end] .^ 2))
error = (1/sample_count) * (cost_part_1 - cost_part_2) + lambda_regul
grad_error_all = ((1/sample_count) * (aug_features') * (hypothesis - outcome)) +
((1/sample_count) * (reg_param * weights))
grad_error_all[1] = ((1/sample_count) * (aug_features[:,1])' * (hypothesis -
outcome))[1]
return(error, grad_error_all)
end

# ╔═╡ 3770eeef-3a77-4e90-861b-aea8efdd0b57
function accuracy(train_norms, model)
    acc = 0
    for (x,y) in train_norms
        @show model(x)      
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(train_norms);
end

# ╔═╡ 37ee477f-ee7a-49d8-8a80-d261ef97c55a
function train_model(loss = loss, acmodel = model, train_set = train_set, test_set = test_set, opt = ADAM(0.001),
     save_name = "convnet.bson")
    best_acc = 0.0
    last_improvement = 0
    training_losses = Vector{Float32}()
    test_losses = Vector{Float32}()
    accuracies = Vector{Float32}()
    for epoch_idx in 1:100

        @eval Flux.istraining() = true
        Flux.train!(loss, params(model), train_set, opt)
        training_loss = sum(loss(train_set[i]...) for i in 1:length(train_set))
        @eval Flux.istraining() = false
        test_loss = loss(test_set...)
        acc = accuracy(test_set...)



        println("Epoch ", epoch_idx,": Training Loss = ", training_loss, ", Test accuracy = ", acc)
        append!(training_losses, training_loss)
        append!(accuracies, acc)
        append!(test_losses, test_loss)

        if acc >= 0.999
            break
        end

        if acc > best_acc
            println("New best accuracy")
            save_name_best = "bson_outputs/"*split(save_name, ".")[1]*"_best.bson"
            BSON.@save save_name_best model epoch_idx accuracies training_loss test_loss
            best_acc = acc
            last_improvement = epoch_idx
        end

        #if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        #    opt.eta /= 10
        #    println("Dropping learn rate to $(opt.eta)")
        #end

        if epoch_idx - last_improvement >= 10
            println("Converged")
            break
        end
    end
    BSON.@save "bson_outputs/"*save_name model accuracies training_losses test_losses
    end


# ╔═╡ ec8c0571-e004-465c-90be-55b4e70e7723
Flux.train!(loss, ps, datasetx, optimizer, cb = throttle(evalcb, 10))

# ╔═╡ 63acd015-4b12-46a8-a8f4-183495d21987
accuracy(train_normals)

# ╔═╡ 992b14f8-9158-4ba1-ab1b-ab5dc5a0cf52
train_images, train_labels, test_images, test_labels = shuffle_data(train_data, train_labels, test_data, test_labels);

# ╔═╡ 9b420c96-4991-4689-a6dd-40359125788e
plot(tr_weights_errors[1],label="Cost",ylabel="Cost",xlabel="Number of Iteration",
title="Cost Per Iteration")

# ╔═╡ ff7785b1-2577-44a4-8309-c5145d5168d8
function get_predictions(features, weights)
total_entry_count = size(features)[1]
aug_features = hcat(ones(total_entry_count, 1), features)
preds = sigmoid(aug_features * weights)
return preds
end

# ╔═╡ f0ce596c-e9f3-490a-92cb-e71826bc5e9b
function get_predicted_classes(preds, threshold)
return preds .>= threshold
end

# ╔═╡ fa08b1fa-6a75-4003-9265-dc5a2dfb921d
train_model()

# ╔═╡ Cell order:
# ╠═e5dd0431-45b4-46c5-8b42-24df6b00f4c0
# ╠═cef296fc-8f9f-411d-882d-854bc6982146
# ╠═66f4116d-8c85-4a17-a4c6-8043ffe7d7cf
# ╠═7421e5e6-d140-491f-b5ac-324a8c0a7661
# ╠═d9ef963a-cb58-4ae2-9cb0-3063f74732b9
# ╠═6ed42c7a-3419-4939-9d8b-60d0ce3fa42a
# ╠═33cbeeb1-3dde-4238-a6bb-ad0ccfc31f29
# ╠═dfcfb664-d1b2-404c-99fd-58733a913b89
# ╠═87f1f15d-0571-4763-a120-0ec65ebc6cd9
# ╠═7a13316d-1a6f-4b0e-aead-55cb0df7b6e5
# ╠═fb740e29-e0fc-4df1-9f6e-b0f188620bec
# ╠═f41ad983-0d1f-44d9-8204-02ed5571a237
# ╠═cd29aa06-1eee-4439-8b14-a63f979dcddb
# ╠═4d29ac13-9012-4049-bd51-5ebe7dc23c83
# ╠═dfc4a3ab-83fa-44b1-bf71-5ede63fcf39a
# ╠═5666d982-1e2b-41e3-96a1-623c0f2beff0
# ╠═a36bf38e-da11-4fa9-b7df-c5c2d62df844
# ╠═a6e51c25-fe10-446e-9506-acafb5dba8a7
# ╠═7c7b3c83-c227-4c83-909b-ce52229bac56
# ╠═a811db25-52c1-47f9-ad13-17bee1664b8b
# ╠═1b6e40f5-e7aa-4985-9c15-85c87316d836
# ╠═27e9d7fa-f8ab-40ce-9a0c-a65c799fa741
# ╠═e624cc4e-cf30-469b-9016-198e87add9c5
# ╠═7f5b9f27-5753-4d5e-aa2e-bf4d4f7cee28
# ╠═cfa3bbf4-fddb-404d-9a8a-03b05bc90f32
# ╠═015c165c-5b5e-40cd-b286-99e2e7c16892
# ╠═68e13bfb-b17e-438d-8f6c-53e99a2926ce
# ╠═911ffd22-c2fa-455d-98c6-46455ddd21bc
# ╠═2ec6642e-3a3f-42a2-abdb-bf2f5bf357d7
# ╠═efd60a88-57a5-42c2-b9ed-a53af6c39bf8
# ╠═695c5380-f5e4-4261-991e-60d91d9d644f
# ╠═1455bf85-05ec-4c16-9bac-e1f30b20c810
# ╠═45448190-523f-4ac3-8e6f-5f81de52c4fb
# ╠═d4801ed9-ca9a-48d5-ba6a-37897dd3659b
# ╠═1e7e4e24-6a0d-439c-8e3a-36fea1bd50c3
# ╠═631b6525-6de7-4e4b-be3a-e640268c3b38
# ╠═6470bd9c-9626-458b-9c4c-df0c058bbaa1
# ╠═b32d08fd-af76-45ce-b3cd-47602dd4a8c6
# ╠═a206d691-57e3-4528-ab0b-7f0579eb2342
# ╠═f41dddef-b173-494c-896e-fc5c591503da
# ╠═ea1130c0-a9c4-4bf8-9c4d-c0c4af22ddca
# ╠═25dc1648-d99a-403e-bfbb-ed7b7cd18ecc
# ╠═e50ab593-0524-4760-aff2-a45ddf094fdf
# ╠═8e04eb33-d25c-45f7-ab14-d484a3a5353c
# ╠═2ae6cf01-1690-4236-960a-e5e9606a3b4d
# ╠═f7cc9fb5-979d-4c57-be94-9aa264385591
# ╠═86c1acc8-81f8-47a7-953c-48f72fec8482
# ╠═1a9ae560-8c95-478c-bc46-0fa49c0eef9a
# ╠═9f72674f-0907-4aec-8861-3cab74479859
# ╠═9eb0d907-5225-4b10-aa0b-d50a8ebf4792
# ╠═06be2f09-0a49-487a-be70-e79ec61554d6
# ╠═6e3f2c68-70f4-466d-b2da-6a5506216799
# ╠═fd930a19-4355-4cad-aefa-73cb1c307517
# ╠═d319751c-9eb7-4cff-8137-bc07bb2f934c
# ╠═13699545-2e1c-4eb1-bbe4-94cb9e770f7c
# ╠═20e5febc-01ed-45cd-bb46-994324ee7262
# ╠═95cce0a0-e2d0-4b81-939d-741b34a39487
# ╠═8031ff2d-3430-4797-9146-1d9b87238fdd
# ╠═39f6a4e4-58e0-4373-a915-1d78ad6b3b35
# ╠═a6862cea-1cb6-4bc9-9b05-e74615b2cd3b
# ╠═9e79aa28-2987-4b89-9c69-d378703d65b5
# ╠═1a70f812-5b5a-4ba8-a966-50cfb957b4b7
# ╠═18cd770a-935f-44eb-9a77-3295442527b2
# ╠═1e669ffb-eba3-49c9-a727-616afbc8c8ca
# ╠═b7ba5c7c-382f-4802-a3fa-d53e8d672979
# ╠═dede71e8-dca7-4380-8681-927092dd58c8
# ╠═5aeee1f4-3712-425a-863d-75cdbecee847
# ╠═b3164ca1-fe99-4b32-8b8f-de16fdcaac33
# ╠═578675d7-136f-4474-ad57-2664e89cd194
# ╠═50bec8e1-e1b1-4903-95b0-b9b87427bb78
# ╠═76f81a2a-8d84-439c-8694-db6b63daf77f
# ╠═a4fb1b74-3647-4712-a85d-1408be03a87c
# ╠═3b015457-bfb6-48d7-9d6d-aa6597b48175
# ╠═9de323c0-e50e-404e-ba43-60ab6a1ddcea
# ╠═c140cddc-e416-4105-90d3-1a80c7db11f8
# ╠═c52730da-bfaa-43fe-bc47-fea5349995e6
# ╠═9557f1c9-7d47-4e85-aa34-9de4489c6fe3
# ╠═e7be8846-d143-457c-880f-06c9846cf0aa
# ╠═c1e8b6b6-09ea-4e6f-86ef-78164e31bad5
# ╠═239fa5df-bd65-49de-8739-7cc90b737983
# ╠═925f4d88-6099-40ba-99f0-000f82a7961a
# ╠═6ee37de2-09cc-44c9-af1f-0c2c6e395823
# ╠═8b530b2a-9599-485d-9c9d-d0091edc70bb
# ╠═a18424f5-b559-4d93-8f3c-2bf08dd23b7c
# ╠═37ee477f-ee7a-49d8-8a80-d261ef97c55a
# ╠═32e21624-33b8-43c5-b036-ac3611817af2
# ╠═3aba7316-e3a0-44f7-867d-0f720d2ad366
# ╠═b8a6f4a3-2aa1-4291-a215-6108943aff88
# ╠═eed53fba-9e06-4204-a67b-88ac47cb61c3
# ╠═73612f7c-26f7-41b2-b390-f054d8af656d
# ╠═db362bbf-9fa8-4b21-a9c5-640c2160e955
# ╠═37f6feaa-ca59-4f64-aabc-b3aa5467aec3
# ╠═f41b32c2-3ae5-47fb-91c6-c612eab11822
# ╠═9052c831-b80b-4222-8ae4-a4c424629047
# ╠═427ccc3f-d926-4b68-9e2c-36ae27a01788
# ╠═5f98aff3-8285-4342-9faf-e520ae995455
# ╠═ccd496fd-c654-4dd9-80fe-70ae8b69d7f6
# ╠═aac22a22-b4d6-4148-ab6c-eab933ade2bd
# ╠═d850ea8e-6a21-407e-a30b-a5f5da9d0393
# ╠═a08cbbcd-19c7-4040-85c7-7bb1fedbea07
# ╠═8e4cedcb-a3cd-49b8-a10e-bb61b13cb661
# ╠═ac2a0b34-eed2-4719-90df-7b3eadc75e0c
# ╠═a815ab6a-6aa2-4691-b77a-e75997ed1696
# ╠═a73a2b9a-23b7-4bef-b3c9-88e5fab0ff5a
# ╠═3770eeef-3a77-4e90-861b-aea8efdd0b57
# ╠═ec8c0571-e004-465c-90be-55b4e70e7723
# ╠═63acd015-4b12-46a8-a8f4-183495d21987
# ╠═992b14f8-9158-4ba1-ab1b-ab5dc5a0cf52
# ╠═9b420c96-4991-4689-a6dd-40359125788e
# ╠═ff7785b1-2577-44a4-8309-c5145d5168d8
# ╠═f0ce596c-e9f3-490a-92cb-e71826bc5e9b
# ╠═fa08b1fa-6a75-4003-9265-dc5a2dfb921d
