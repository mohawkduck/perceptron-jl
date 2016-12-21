type NeuralNetwork
  layer_neuron_nums::Array{Number,1}
  input_num::Number
  output_num::Number
  length_hidden_layers::Number
  length_layers::Number
  bias::Number
  learningRate::Number
  neurons_act_values::Array{Array{Number},1}
  neurons_grad_values::Array{Array{Number},1}
  neurons_weights::Array{Array{Number},1}
  function NeuralNetwork()
    new([],0,0,0,0,0,0,[],[],[])
  end
end

function initNeuralNet(m::NeuralNetwork,input_num, output_num, length_hidden_layers)
  m.input_num = input_num
  m.output_num = output_num
  m.length_hidden_layers = length_hidden_layers
  m.bias = 1
  m.learningRate = 0.15

  # 바이어스를 추가한 뉴런 개수를 레이어별로 계산한다.
  m.length_layers = m.length_hidden_layers + 2
  for l in 1:(m.length_layers - 1)
    push!(m.layer_neuron_nums, (m.input_num + 1))
  end
  push!(m.layer_neuron_nums, (m.output_num + 1))

  # 레이어별로 activation 결과를 저장하는 배열이다.
  m.neurons_act_values = [[0 for i in 1:m.layer_neuron_nums[j]] for j in 1:m.length_layers]
  for act_values in m.neurons_act_values
    act_values[length(act_values)] = m.bias
  end

  # 레이어별로 학습시킬 gradient를 저장하는 배열이다.
  m.neurons_grad_values = [[0 for i in 1:m.layer_neuron_nums[j]] for j in 1:m.length_layers]

  # weight를 초기화하고 랜덤값을 입력한다.
  m.neurons_weights = [[i+j for i=1:length(m.neurons_act_values[k+1]), j=1: length(m.neurons_act_values[k])] for k in 1:(m.length_layers - 1)]
  for i in 1:length(m.neurons_weights)
    m.neurons_weights[i] = rand(size(m.neurons_weights[i],1),size(m.neurons_weights[i],2))
  end
end

function getAvtivate(m::NeuralNetwork, x)
  # activation 함수는 ReLU를 사용한다.
  max(0.0, x)
end

function getActGrad(m::NeuralNetwork, x)
  if x > 0.0
    return 1.0
  else
    return 0.0
  end
end

function feedForward(m::NeuralNetwork)
  # 각 레이어의 뉴런를 실행해서 값을 출력한다.
  for i in 1:length(m.neurons_weights)
    m.neurons_act_values[i+1] = m.neurons_weights[i] * m.neurons_act_values[i]
    for j in 1:length(m.neurons_act_values[i+1])-1
      m.neurons_act_values[i+1][j] = getAvtivate(m,m.neurons_act_values[i+1][j])
    end
  end
end

function feedBackward(m::NeuralNetwork, target)
  # calculate gradients of output layer
  # 학습 target값과 출력값의 차이를 출력 레이어 gradient에 입력한다.
  l = length(m.neurons_grad_values)
  for d in 1:length(m.neurons_grad_values[l])-1
    m.neurons_grad_values[l][d] = (target[d] - m.neurons_act_values[l][d]) * getActGrad(m,m.neurons_act_values[l][d])
    # skips last component (bias)
  end

  # calculate gradients of hidden layers
  # hidden layer에 들어갈 학습값을 계산한다.
  for l in length(m.neurons_weights):-1:2
    m.neurons_grad_values[l] = transpose(m.neurons_weights[l]) * m.neurons_grad_values[l+1]
    for d in 1:length(m.neurons_grad_values[l])-1
      m.neurons_grad_values[l][d] = m.neurons_grad_values[l][d] * getActGrad(m,m.neurons_act_values[l][d])
    end
  end

  # update weights after all gradients are calculated
  # 계산한 학습값을 weight에 적용시킨다.
  for l in length(m.neurons_weights):-1:1
    for row in 1:size(m.neurons_weights[l],1)
      for col in 1:size(m.neurons_weights[l],2)
        m.neurons_weights[l][row, col] = m.neurons_weights[l][row, col] + (m.learningRate * m.neurons_grad_values[l+1][row] * m.neurons_act_values[l][col])
      end
    end
  end
end

function setInputs(m::NeuralNetwork, inputs)
  for i in 1:m.input_num
    m.neurons_act_values[1][i] = inputs[i]
  end
end

function getOutputs(m::NeuralNetwork)
  output_value = []
  for i in 1:m.output_num
    append!(output_value, m.neurons_act_values[length(m.neurons_act_values)][i])
  end
  return output_value
end

function main()
  nn = NeuralNetwork()
  initNeuralNet(nn,2,1,1)
  #input_x = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
  #train_y = [[0],[1],[1],[0]]
  #현재 정확한 원인을 알수없으나 XOR 학습이 잘 안된다.
  #예제 C++ 코드에서 정상작동 하는 것으로 보인다. 변수가 overflow되거나 값이 씹히는 것 같다. 추적해서 알아낸다.

  input_x = [[1.0,1.0]]
  train_y = [[3]]
  for i in 1:100 # 학습 횟수 100회
    for j in 1:length(input_x)
      setInputs(nn,input_x[j])
      feedForward(nn)
      feedBackward(nn,train_y[j])
    end
  end

  # 학습시킨대로 출력이 나오는지 테스트한다.
  for j in 1:length(input_x)
    setInputs(nn,input_x[j])
    feedForward(nn)
    println("*outputs = ",getOutputs(nn))
    println("***act ",nn.neurons_act_values)
    println("***grad ",nn.neurons_grad_values)
    println("***weights ",nn.neurons_weights)
  end
end
main()
