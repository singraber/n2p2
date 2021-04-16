var classnnp_1_1NeuralNetwork =
[
    [ "Layer", "structnnp_1_1NeuralNetwork_1_1Layer.html", "structnnp_1_1NeuralNetwork_1_1Layer" ],
    [ "Neuron", "structnnp_1_1NeuralNetwork_1_1Neuron.html", "structnnp_1_1NeuralNetwork_1_1Neuron" ],
    [ "ActivationFunction", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbed", [
      [ "AF_IDENTITY", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbeda5b0c59d80a0e661ef1f0bfc3df1f5a04", null ],
      [ "AF_TANH", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbeda1569427fccd98a9f3e8ec25d334b982b", null ],
      [ "AF_LOGISTIC", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbeda5915a5141aaee3c8fd8ea61da99b3b41", null ],
      [ "AF_SOFTPLUS", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbeda85811bbf7c259d1fee628b2a69114530", null ],
      [ "AF_RELU", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbedafe93c3a968224442e85a58ba3627e3a9", null ],
      [ "AF_GAUSSIAN", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbeda66545b5c028ebb8934093188e3279492", null ],
      [ "AF_COS", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbeda2721d2639f95e923ac6fafd0980a2588", null ],
      [ "AF_REVLOGISTIC", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbedaf1b2dce761bdf00c9ff8d6ebd1374578", null ],
      [ "AF_EXP", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbedad56bdaa89dd2aa1d72d539995267a967", null ],
      [ "AF_HARMONIC", "classnnp_1_1NeuralNetwork.html#a032b3b525f06cd70953aec8e6aeedbedaf565187036a46ee35d905df14542dc01", null ]
    ] ],
    [ "ModificationScheme", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1", [
      [ "MS_ZEROBIAS", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1a66ae2f43aca0c9fbc5285c4b81e79bfa", null ],
      [ "MS_ZEROOUTPUTWEIGHTS", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1a327dc589d3a961565fb0754bad7b80f3", null ],
      [ "MS_FANIN", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1aa284452e4756a63d3fe791ea6739ab08", null ],
      [ "MS_GLOROTBENGIO", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1ab96a1476ad0c41c52f782708a82d7949", null ],
      [ "MS_NGUYENWIDROW", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1ade647e2d9f5c0d6f8f20d95df165c827", null ],
      [ "MS_PRECONDITIONOUTPUT", "classnnp_1_1NeuralNetwork.html#a87219d930829f343d65e5c59bc0220b1a9c1def9ad2311fe4774afdc8790406ce", null ]
    ] ],
    [ "NeuralNetwork", "classnnp_1_1NeuralNetwork.html#ad8b7ab63576027f56f1cbae20d2d7d58", null ],
    [ "~NeuralNetwork", "classnnp_1_1NeuralNetwork.html#a65475a7d7b05d302392333302626b2f8", null ],
    [ "setNormalizeNeurons", "classnnp_1_1NeuralNetwork.html#aa3970c1a57ef211ce23e5dfd85619e93", null ],
    [ "getNumNeurons", "classnnp_1_1NeuralNetwork.html#a0f1499117239804e1e33e20234021eef", null ],
    [ "getNumConnections", "classnnp_1_1NeuralNetwork.html#a0a5977b23676166a39fefbb0d232f050", null ],
    [ "getNumWeights", "classnnp_1_1NeuralNetwork.html#a2f31e2bf59c2cb5e378a40d7caa5077e", null ],
    [ "getNumBiases", "classnnp_1_1NeuralNetwork.html#aa3e377047279b8b0bbd11c80a8aefb3d", null ],
    [ "setConnections", "classnnp_1_1NeuralNetwork.html#a3479cb0b5dc64c4260837adbfa11f868", null ],
    [ "getConnections", "classnnp_1_1NeuralNetwork.html#a93e137591120925fe9ec70011385a972", null ],
    [ "initializeConnectionsRandomUniform", "classnnp_1_1NeuralNetwork.html#a5f3200be3d54c5960bc979a1ba27e9d7", null ],
    [ "modifyConnections", "classnnp_1_1NeuralNetwork.html#a12b98d35abbe2cd5d6357e061a2a5bc7", null ],
    [ "modifyConnections", "classnnp_1_1NeuralNetwork.html#ad09333ff42f79590846ab2af21e37fcb", null ],
    [ "setInput", "classnnp_1_1NeuralNetwork.html#a0ddcfb0243cca5956075a75c90431c37", null ],
    [ "setInput", "classnnp_1_1NeuralNetwork.html#a358401db86795aa046101e18db0e12c6", null ],
    [ "getOutput", "classnnp_1_1NeuralNetwork.html#ad7e18949cd215e5427440fce7a1437d7", null ],
    [ "propagate", "classnnp_1_1NeuralNetwork.html#a9d2a2d332c23876935e3a3272f40f5c2", null ],
    [ "calculateDEdG", "classnnp_1_1NeuralNetwork.html#abf46d0d0d2a4c068f67bb13ae445d725", null ],
    [ "calculateDEdc", "classnnp_1_1NeuralNetwork.html#aa38d991173328eb5c977dacf39347dca", null ],
    [ "calculateDFdc", "classnnp_1_1NeuralNetwork.html#a4526b9a7e947a38b861098d0826a6161", null ],
    [ "writeConnections", "classnnp_1_1NeuralNetwork.html#a38e5d3edf8eb6c1a28cfddeb674207e2", null ],
    [ "getNeuronStatistics", "classnnp_1_1NeuralNetwork.html#a78f2cbfa7dc6b7879081ed213662453f", null ],
    [ "resetNeuronStatistics", "classnnp_1_1NeuralNetwork.html#a86ead163e79c0680db2355733ef293b7", null ],
    [ "getMemoryUsage", "classnnp_1_1NeuralNetwork.html#a6163a65341980f16ec4376ec035f609f", null ],
    [ "info", "classnnp_1_1NeuralNetwork.html#aee07252d0a6b47e550117120470845f3", null ],
    [ "calculateDEdb", "classnnp_1_1NeuralNetwork.html#a2ed910e1e6cd8ba8ffa9bb495ea96656", null ],
    [ "calculateDxdG", "classnnp_1_1NeuralNetwork.html#acc168a80af0a840675f46ed31c7a6723", null ],
    [ "calculateD2EdGdc", "classnnp_1_1NeuralNetwork.html#a7619d39464dd49cbfedd1ec6d86fe71b", null ],
    [ "allocateLayer", "classnnp_1_1NeuralNetwork.html#a4f06892244af00ab70125061bbcbe3c3", null ],
    [ "propagateLayer", "classnnp_1_1NeuralNetwork.html#a45b217aee588e3a9d443f497e4cc5bf4", null ],
    [ "normalizeNeurons", "classnnp_1_1NeuralNetwork.html#a41aedf8cb1f2f786205c7d792870077b", null ],
    [ "numWeights", "classnnp_1_1NeuralNetwork.html#ad7ca98e12a802d108ad987acd3beb088", null ],
    [ "numBiases", "classnnp_1_1NeuralNetwork.html#a7924eda7794c924a52474b45f83acfb8", null ],
    [ "numConnections", "classnnp_1_1NeuralNetwork.html#a964d98ad76e025e3f95c87e8fa322d27", null ],
    [ "numLayers", "classnnp_1_1NeuralNetwork.html#a1f2248da5de33a697f76780423ecc3be", null ],
    [ "numHiddenLayers", "classnnp_1_1NeuralNetwork.html#ae71395d9b5b66ad73c8f1f2eac764aa9", null ],
    [ "weightOffset", "classnnp_1_1NeuralNetwork.html#a59be6ffbe6c6c5ae246cb105e188fa1b", null ],
    [ "biasOffset", "classnnp_1_1NeuralNetwork.html#a6e9ce480278a211b224a135a16a9efe6", null ],
    [ "biasOnlyOffset", "classnnp_1_1NeuralNetwork.html#a0f8a213775aa8f185e85748f0bbf4a79", null ],
    [ "inputLayer", "classnnp_1_1NeuralNetwork.html#a0c17ca3a07ca010d5911a8fb93a71069", null ],
    [ "outputLayer", "classnnp_1_1NeuralNetwork.html#ad88ef411d3514a71f20a880f0ba81336", null ],
    [ "layers", "classnnp_1_1NeuralNetwork.html#ad39acdd5f3f75914a8ed4bb28dc1f56a", null ]
];