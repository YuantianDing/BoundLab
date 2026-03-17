
$$
\begin{aligned}
x_0 = y_0 = \epsilon \in [-1, 1]^n \\
x_1 = \text{relu}(W_1 @ x_0 + b) = \text{relu}(W_1 @ \epsilon + b) \\
y_1 = \text{relu}(W_2 @ y_0 + b) = \text{relu}(W_2 @ \epsilon + b)
\end{aligned}
$$

$$
\begin{aligned}
l_x = b - \|W_1 \textbf{1}_n\| \leq W_1 @ \epsilon + b \leq b + \|W_1\textbf{1}_n\| = u_x \\
l_y = b - \|W_2 \textbf{1}_n\| \leq W_2 @ \epsilon + b \leq b + \|W_2\textbf{1}_n\| = u_y \\
\end{aligned}
$$

$$
\begin{aligned}



