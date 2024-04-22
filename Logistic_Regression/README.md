# Logistic Regression

## Nhìn lại các mô hình tuyến tính
- Nhìn lại các mô hình tuyến tính trước đó, chúng ta thấy các mô hình đều có công thức chung là:
$$
    y = f(w^Tx)
$$

- Thực tế, mô hình Linear Regression có dạng $y = w^Tx$ và Perceptron lại có dạng $y = sgn(w^Tx)$. Lúc này chúng ta cần một cái tên gọi chung cho hàm f và họ gọi nó là *activation function*.

## Giới thiệu về Neural Network

- Mạng neural (neural network) là một mô hình toán học được lấy cảm hứng từ cách hoạt động của não người. Nó bao gồm một tập hợp các đơn vị tính toán được gọi là "neuron" hoặc "nơ ron" được kết nối với nhau và xử lý dữ liệu.

![](/Logistic_Regression/Image/Neural-Networks-Architecture.png)

- **Input layer**: Lớp này nhận dữ liệu đầu vào và truyền nó qua mạng.
- **Hindden layer**: Đây là các lớp nằm giữa lớp đầu vào và lớp đầu ra. Mỗi neuron trong các lớp ẩn nhận đầu vào từ các neuron trong lớp trước đó, tính toán và truyền đầu ra của mình cho các neuron trong lớp tiếp theo.
- **Output layer**: Lớp này đưa ra đầu ra của mạng sau khi dữ liệu đã được xử lý qua các lớp ẩn.
- Mỗi một mô hình neural network có thể có hoặc không có các hidden layer, tổng số layer cho một mô hình sẽ không bao gồm input layer, trong hình trên thì mạng neuron sẽ có 3 lớp.
- Các hình tròn là các node, tại mỗi node chúng ta sẽ thực hiện:
    - Liên kết với tất cả các node ở layer trước đó với các hệ số w riêng.
    - Mỗi node có một hệ số bias riêng.
    - Diễn ra hai bước, tính tổng linear và áp dụng activation function.

![](/Logistic_Regression/Image/pla_nn.png)

- Nhìn lại bài trước, trên đó hình ảnh diễn tả hàm xác định label của percepron $y = sgn(w^Tx)$.

- Hình trên chỉ biểu diễn hình của neural network không có hindden layer.

- Hàm $y = sgn(w^Tx)$ chính là activation function của bài toán.

## Activation Function

![](/Logistic_Regression/Image/alll_activation_funcs.png)

- Activation Function là một phần quan trọng trong mạng neu ron, có tác dụng tăng tính phi tuyến, tính toán đầu ra của một mạng neu ron, giúp cho mô hình hiểu và học được các dữ liệu phức tạp.

![](/Logistic_Regression/Image/activation_func.gif)
 
- Lý do chúng ta cần sử dựng activation function:
    - Đối với các mô hình có dữ liệu phức tạp thì lúc này một mô hình phi tuyến sẽ không phát huy được tác dụng. Hơn nữa, nó cũng không giúp chúng ta chặn được các khoảng đầu ra, khiến chúng ta tốn thời gian train mô hình mà không học hiệu quả các feature của dữ liệu.

- Đọc thêm về Activation Function: [Activation Function Guide ](https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/)

## Mô hình logistic regression
- **Logistic Regression** là thuật toán ước lượng xác suất để một sự kiện xảy ra, dựa trên một tập dữ liệu gồm các biến phụ thuộc.
- Mặc dù trong tên có *regression* nhưng logistic regression lại là bài toán đặc thù để xử lý các vấn đề classification. Lấy theo ví dụ của MachineLearningCoBan thì chúng ta cần tạo ra một mô hình để dự đoán tỷ lệ đỗ đạt dựa trên số giờ học của học sinh.

![](/Logistic_Regression/Image/ex1.png)

Hình dưới đây là một số activation function của các thuật toán tuyến tính.

![](/Logistic_Regression/Image/activation_funcs.png)

- Hãy thử phân tích lại độ phù hợp với bài toán của các đường này nhé:
    - Đầu tiên là Linear Regression, cơ bản thuật toán này chịu sự ảnh hưởng rất lớn bởi nhiễu. Điều này khiến cho mô hình không được chính xác khi dữ liệu huấn luyện xuất hiện nhiễu. Hơn nữa, đầu ra của thuật toán này không bị chặn khoảng từ 0 đến 1.

    ![](/Logistic_Regression/Image/ex1_lr.png)

    - Thứ hai là ngưỡng cứng (hard thresold), có thể nhận ra đây chính là độ thị hàm $y = sgn(w^Tx)$ trong thuật toán Perceptron, mà Perceptron chỉ phù hợp với bài toán có dữ liệu là linear separable thôi.

    - Hai đường còn lại phù hợp hơn với bài toán, khi nó mang tính chất:
        - Được chặn khoảng từ (0, 1)
        - Liên tục, có đạo hàm tại mọi điểm, có thể áp dụng GD.

- Như đã nói ở trên, đối với các mô hình tuyến tính chúng ta đều coi nó là các activation function, bây giờ chúng ta cần tìm một activation function sao cho phù hợp với dữ liệu bài toán logistic regression, đảm bảo được các tính chất của hai đường thẳng trên.

- Công thức tổng quát cho logistic regression:

$$
f(s) = \frac{1}{1 + e^{-s}} \equiv \sigma(s)
$$

## Hàm mất mát của Logistic Regression
- Đối với bài toán Logistic Regression, hàm loss được xây dựng chính là Cross Entropy Loss.

- Cross Entropy Loss thường được sử dụng để đo khoảng cách giữa hai phân phối. Khoảng cách giữa hai phân phối nhỏ đồng nghĩa với việc là hai phân phối đó sẽ gần nhau.

$$
J(w)  = -\sum_{i=1}^{N} \left( y_i \log z_i + (1 - y_i)\log (1 - z_{i}) \right)
$$

- Cách xây dựng hàm mất mát và cách tối ưu: [Cross Entropy Loss and Optimization Method](https://machinelearningcoban.com/2017/01/27/logisticregression/#-ham-mat-mat-va-phuong-phap-toi-uu)


## Công thức cập nhật trọng số
$$
w = w + \eta \left( y_i - z_i \right) x_i
$$

