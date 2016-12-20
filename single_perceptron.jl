type Neuron
  w::Number
  b::Number
  input::Number
  output::Number
  Neuron() = new(0.0,1.0,1.0,1.0)
end

function getActivation(m::Neuron, target::Number)
  # return max(0, target) # ReLU
  return target
end

function getActivationGradiation(m::Neuron, target::Number)
  return 1.0
end

function feedForward(m::Neuron)
  m.output = getActivation(m, m.w*m.input + m.b)
end

function propagateBacward(m::Neuron, target::Number)
  learningRate = 0.1
  m.w = m.w - learningRate * (m.output - target) * getActivationGradiation(m,m.output) * m.input
  m.b = m.w - learningRate * (m.output - target) * getActivationGradiation(m,m.output)
end

function main()
  neuron = Neuron()
  for i in 1:100
    neuron.input = 1.0
    output = feedForward(neuron)
    propagateBacward(neuron, 4.0)
    println(i, " to ", output)
  end
  println("weight ", neuron.w)
end

main()
