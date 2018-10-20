
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

# e-greedy
plot1, = plt.plot(
	[1/64, 1/32, 1/16, 1/8, 1/4],
	[1.2332039927871679, 1.3510964751538432, 1.3969254689491928, 1.3756075883697045, 1.2594456293233878]
)

# e-greedy and alpha
plot2, = plt.plot(
	np.array([1/64, 1/32, 1/16, 1/8, 1/4]),
	[1.3932540648541343, 1.3994634403448867, 1.3984898123010694, 1.3635468934825767, 1.2458611432508624]
)

# e-greedy, alpha and initial
plot3, = plt.plot(
	np.array([1/128, 1/64, 1/32, 1/16]),
	[1.5114941500085874, 1.5045500670594925, 1.490420028255835, 1.4586311972969734],
)

# UCB
plot4, = plt.plot(
	np.array([1/4, 1/2, 1, 2, 4]),
	[1.5226647954797536, 1.5305905557802615, 1.5176101323132476, 1.440517813415161, 1.2494235051618803]
)

# Gradient
plot5, = plt.plot(
	[1/16, 1/8, 1/4, 1/2],
	[1.3543108410700546, 1.4476678343099378, 1.4896252915969828, 1.4736398865903912]
)

plt.xscale('log')
plt.xticks([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4], [Fraction(1, 128), Fraction(1, 64), Fraction(1, 32), Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1,2), 1, 2, 4])

plt.legend(
	(plot1, plot2, plot3, plot4, plot5),
	( r'$\epsilon$-greedy', r'$\epsilon$-greedy with $\alpha$', r'$\epsilon$-greedy with $\alpha$ and initial', 'UCB', 'Gradient')
)

plt.xlabel(r'$\epsilon$, c')

plt.ylabel('Average return')

plt.title('Multiarmed Bandits')

plt.savefig('graph.png')

plt.show()
