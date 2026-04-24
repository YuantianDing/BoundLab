softmax2(x, y) = x / (1 + x exp(y)) 
               = exp(-y) (1 - 1 / (1 + x exp(y))
softmax2dx(x, y) = 1 / (1 + x exp(y))^2
softmax2dx-1(lamx, y) = (1/sqrt(lamx) - 1) / exp(y)
softmax2dy(x, y) = x^2 exp(y) / (1 + x exp(y))^2
                 = softmax2(x, y) x exp(y) / (1 + x exp(y))
                 = softmax2(x, y) (1 - softmax2(x, y)/x)
softmax2dy-1+(x, lamy) = x/2 (1 + sqrt(1 - 4 lamy / x))
softmax2dy-1-(x, lamy) = x/2 (1 - sqrt(1 - 4 lamy / x))

softmax2maxx(lamx, lx, ux, y)
    = arg-max (lx ≤ x ≤ ux) softmax2(x, y) + lamx x
    = clamp(softmax2dx-1(lamx, y), lx, ux)
softmax2minx(lamx, lx, ux, y)
    = arg-min (lx ≤ x ≤ ux) softmax2(x, y) + lamx x
    = with lam0 = (softmax2(ux, y) - softmax2(lx, y)) / (ux - lx)
        if lamx ≤ lam0 then lx else ux
softmax2minx(lamx, lamy, lx, ux, ly, uy)
    = arg-min (lx ≤ x ≤ ux, ly ≤ y ≤ uy) softmax2(x, y) + lamx x + lamy y
    = 



        




