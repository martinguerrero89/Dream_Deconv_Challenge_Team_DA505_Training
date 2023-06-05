#Rhino_norm

mynorm4= function(l){
  lr= rank(l*-1,ties.method="max")
  lr=log2(lr)
  lr= lr*-1
  lr= lr+ log2(length(lr))
  res=lr
  return(res)
}
