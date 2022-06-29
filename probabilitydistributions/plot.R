curve(dnorm, -3.5, 3.5, lwd=2, axes = FALSE, xlab = "", ylab = "")
axis(1, at = -3:3, labels = c("-3考", "-2考", "-1考", "mean", "1考", "2考", "3考"))
polygon(c(-1,x,1),c(0,y,0),col="blue")


# Normal distribution - same mean
x  <- seq(0, 8, 0.01)

plot(x, dnorm(x, mean = 4, sd = 0.5), type = 'l', ylab = "", xlab = "", col = 'red', ylim = c(0, 1))
lines(x, dnorm(x, mean = 4, sd = 1), type = 'l', col = 'blue')
lines(x, dnorm(x, mean = 4, sd = 2), type = 'l', col = 'green')

title('Normal distribution')

legend(5, 1, legend=c(expression(paste(mu == 4, ",  ", sigma == 0.5, "       ")),
                      expression(paste(mu == 4, ",  ", sigma == 1)),
                      expression(paste(mu == 4, ",  ", sigma == 2))), 
       col=c("red", "blue", "green", "orange"), lty=1, cex = 0.8)


# Normal distribution
x  <- seq(-5, 5, 0.01)

plot(x, dnorm(x, mean = 0, sd = sqrt(0.2)), type = 'l', ylab = "", xlab = "", col = 'red', ylim = c(0, 1))
lines(x, dnorm(x, mean = 0, sd = sqrt(1)), type = 'l', col = 'blue')
lines(x, dnorm(x, mean = 0, sd = sqrt(5)), type = 'l', col = 'green')
lines(x, dnorm(x, mean =-2, sd = sqrt(0.5)), type = 'l', col = 'orange')

title('Normal distribution')

legend(2, 1, legend=c(expression(paste(mu == 0, ",  ", sigma^2 == 0.2, "       ")),
                        expression(paste(mu == 0, ",  ", sigma^2 == 1)),
                        expression(paste(mu == 0, ",  ", sigma^2 == 5)),
                        expression(paste(mu == -2, ",  ", sigma^2 == 0.5))), 
       col=c("red", "blue", "green", "orange"), lty=1, cex = 0.7)


# Gamma distribution
x  <- seq(0, 10, 0.01)

plot(x, dgamma(x, shape = 1, rate = 1/2), type = 'l', ylab = "", xlab = "", col = 'red')
lines(x, dgamma(x, shape = 2, rate = 1/2), type = 'l', col = 'blue')
lines(x, dgamma(x, shape = 3, rate = 1/2), type = 'l', col = 'green')

title('Gamma distribution')

legend(6.5, 0.5, legend=c(expression(paste(alpha == 1, ",  ", beta == 0.5, "    ")),
                        expression(paste(alpha == 2, ",  ", beta == 0.5)),
                        expression(paste(alpha == 3, ",  ", beta == 0.5))), 
                        col=c("red", "blue", "green"), lty=1, cex = 0.9)
  

# Inverse Gamma distribution
library(invgamma)
x  <- seq(0, 5, 0.01)

plot(x, dinvgamma(x, shape = 1, rate = 1), type = 'l', ylab = "", xlab = "", col = 'red', ylim = c(0, 5))
lines(x, dinvgamma(x, shape = 2, rate = 1), type = 'l', col = 'blue')
lines(x, dinvgamma(x, shape = 3, rate = 1), type = 'l', col = 'green')
lines(x, dinvgamma(x, shape = 3, rate = 0.5), type = 'l', col = 'orange')

title('Inverse Gamma distribution')

legend(3.2, 5, legend=c(expression(paste(alpha == 1, ",  ", beta == 1, "       ")),
                          expression(paste(alpha == 2, ",  ", beta == 1)),
                          expression(paste(alpha == 3, ",  ", beta == 1)),
                          expression(paste(alpha == 3, ",  ", beta == 0.5))), 
       col=c("red", "blue", "green", "orange"), lty=1, cex = 0.9)
