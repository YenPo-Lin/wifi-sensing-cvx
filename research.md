## 研究動機

人體運動會在 CSI power 中產生不同的 Doppler modulation。若直接對整段 dynamic signal 做空間–頻率估計，不同時間、不同 Doppler component 可能混在一起，降低後續 Azi–ToF 的可解釋性與穩定性。

## 問題定義

* 如何從 CSI power 中辨識特定 target Doppler 的有效時間區段
* 提取該 Doppler 對應的 RX–SC response，進行 Azi–ToF estimation (temporal aware spatial feature map)

## 解決方法

先在 temporal domain 分離 motion component，再做 spatial–frequency estimation

避免不同 Doppler、不同活動時間直接混進同一個 covariance。
因此最終可以得到：

$$ f_{d,1} \rightarrow P(\theta,\tau\mid f_{d,1}) $$ $$ f_{d,2} \rightarrow P(\theta,\tau\mid f_{d,2}) $$

而不是只有一張混合所有動態成分的 Azi–ToF heatmap。

## CSI Model
* $ n $: Packet index
* $ m $: RX index
* $ k $: SC index
* $ r $: Local fitting window index
* $ \ell $: Target Doppler index

$$ H_{m,k}[n] = \sum_{p=1}^{P} \alpha_p e^{-j2\pi \frac{m d\sin\theta_p}{\lambda_c}} e^{-j2\pi k\Delta f\,\tau_p} e^{j2\pi f_{D,p}t_n} + w_{m,k}[n]  $$

除了 Doppler 以外的項全部合併成一個 complex coefficient：

$$ \beta_{m,k,p} = \alpha_p e^{-j2\pi \frac{m d\sin\theta_p}{\lambda_c}} e^{-j2\pi k\Delta f\,\tau_p} $$

那 CSI model 就可以簡化成：

$$ \boxed{ H_{m,k}[n] = \sum_{p=1}^{P} \beta_{m,k,p} e^{j2\pi f_{D,p}t_n} + w_{m,k}[n] } $$

這裡：
$$ \boxed{ \beta_{m,k,p} = \text{path gain + Azi phase + ToF phase} } $$

## Stage 1：Target-Doppler temporal detection

### Derivation
從 dynamic power model：

$$ \widetilde P_{m,k}[n] \approx 2\Re\left\{ H_{s,m,k}^{*}\beta_{m,k,d} e^{j2\pi f_Dt_n} \right\}. $$

簡化： $m^{th} $ RX, $k^{th}$ SC
$$ \widetilde P[n] \approx 2\Re\left\{ H_{s}^{*}\beta_{d} e^{j2\pi f_Dt_n} \right\}. $$
令

$$ H_s^*\beta_d=Ae^{j\phi}, $$

則：

$$ \widetilde P[n] \approx 2A\cos(2\pi f_Dt_n+\phi). $$

問題是：已知 $f_d$ ，但不知道每個 channel 的

$$ A,\qquad \phi. $$

利用

$$ \cos(\omega t+\phi) = \cos\phi\cos\omega t - \sin\phi\sin\omega t, $$

就可以寫成：

$$ \boxed{ \widetilde P[n] = a\cos(2\pi f_Dt_n) + b\sin(2\pi f_Dt_n) +e[n] } $$


### Work
對每個 RX–subcarrier channel 的 dynamic power，在 sliding window 中針對 target Doppler 小範圍做 sine/cosine fitting：

$$ y_c(t) \approx a_c\cos(2\pi f_dt)+b_c\sin(2\pi f_dt). $$

利用 Design matrix：

$$ X= \begin{bmatrix} 1&q_0&\cos(2\pi f_dt_0)&\sin(2\pi f_dt_0)\\ 1&q_1&\cos(2\pi f_dt_1)&\sin(2\pi f_dt_1)\\ \vdots&\vdots&\vdots&\vdots \end{bmatrix} $$

LS Solution:

$$ \begin{bmatrix} \hat a\\ \hat b \end{bmatrix} = (X^TWX)^{-1}X^TWy. $$

其中

$$ R=\sqrt{a^2+b^2} $$


$$ Q_{\ell, r, c} = [ 1-\frac{SSE_{1, \ell, r, c}}{SSE_{0, \ell, r,c}+\epsilon} ]_+$$

* $W$ :Weighted window，讓 window 中央權重高，兩側 samples 權重低
* $ M_0​:y[q]=\beta_0​+\beta_1​q+e[q]​ $
* $ M_1​:y[q]=\beta_0​+\beta_1​q+a\cos(2\pi f_dt)+b\sin(2\pi f_dt)+e[q]​ $
* $ \beta_0 $: 這個 window 自己估出的 DC offset
* $ \beta_1 $: 這個 window 自己估出的 linear trend
* $a ,b$：target Doppler 的 coefficient
* $ e[q] $: 剩下無法解釋的誤差
* SSE: sum of square errors

如果前面已經做了 moving-average subtraction，使得 DC 幾乎被移掉，可以不用很強調 trend。

利用 fitting strength $R$ 與 fitting quality $Q$，找出 **target $f_d$ 真正 active 的時間區段**。

## Stage 2：Doppler-conditioned projection

在 Stage 1 選出的 active interval 中，將每個 channel 的原始 dynamic power 投影到 target Doppler：

$$ z_c^P(f_d) = \sum_n h[n]\widetilde P_c[n]e^{-j2\pi f_dt_n}. $$

得到：

$$ \boxed{ \mathbf Z^P(f_d) \in\mathbb C^{N_{RX}\times N_{SC}} } $$

也就是該 Doppler 對應的 spatial–frequency coefficient map。

再用它建立 covariance，進行：

$$ \boxed{ P(\theta,\tau\mid f_d) } $$

的 Azi–ToF estimation。