--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.alpha'             : smoothing constant
- 'config.epsilon'           : value with which to initialise m
- 'config.weightDecay'       : weight decay
- 'state'                    : a table describing the state of the optimizer;
                               after each call the state is modified
- 'state.m'                  : leaky sum of squares of parameter gradients,
- 'state.tmp'                : and the square root (with epsilon smoothing)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function optim.rmsprop(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-2
   local alpha = config.alpha or 0.99
   local epsilon = config.epsilon or 1e-8
   local wd = config.weightDecay or 0
   local mfill = config.initialMean or 0

   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) initialize mean square values and square gradient storage
   if not state.m then
      state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end

   -- (4) calculate new (leaky) mean squared values
   state.m:mul(alpha)
   state.m:addcmul(1.0-alpha, dfdx, dfdx)

   -- (5) perform update
   state.tmp:sqrt(state.m):add(epsilon)
   x:addcdiv(-lr, dfdx, state.tmp)

   -- return x*, f(x) before optimization
   return x, {fx}
end
