# GRADIENT DESCENT

**GRADIENT DESCENT VISUALIZATION:** [DEEPLEARNING.AI](https://www.deeplearning.ai/ai-notes/optimization/index.html)

## Ôn tập các khái niệm cơ bản

![](/Gradient_Descent/gradient_descent.png)

- Cực trị của hàm số là giá trị khiến cho hàm số đổi chiều khi biến thiên.
- Điểm cực tiểu là một điểm trên đồ thị của một hàm số mà giá trị của hàm số tại điểm đó nhỏ nhất so với các giá trị xung quanh. Trong học máy, chúng ta sẽ sử dụng thuật ngữ local minimum thay thế cho "điểm cực tiểu".
- Điểm có giá trị nhỏ nhất là một local minimum đặc biệt, và được gọi là global minimum.
- Điểm local minimum sẽ khiến cho đạo hàm của hàm số  bằng 0, điểm global minimum sẽ khiến cho hàm số đạt giá trị nhỏ nhất.

- Đối với hình trên, chúng ta có global minimum là điểm x = 1, với giá trị nhỏ nhất là -2

## Motivations
- Việc tìm  weight sao cho loss function là minimum có thể được thực hiện theo hai cách phổ biến như sau:
    - Tìm nghiệm của đạo hàm, nhưng đa số cách này trở nên bất khả thi trong đa số các trường hợp vì đạo hàm có thể có rất nhiều nghiệm (local minimum) nên việc tìm ra giá trị nhỏ nhất (global minimum) sẽ rất khó khăn, một lý do khác là đạo hàm có thể vô nghiệm hoặc rất phức tạp để tính toán.

    - VD: Tìm cực trị cho hàm số sau
    $$
        f(x)= x^3 − 6x^2 + 11x − 6
    $$

    -  Gradient Descent, đây là ý tưởng lấy từ việc học của con người. Ý tưởng cơ bản của thuật toán là chọn một điểm mà chúng ta là gần với nghiệm của bài toán, sau đó dùng phép toán lặp để tiến dần đến điểm cần tìm, tức đến gần gradient bằng 0.

## Gradient Descent
- **Gradient Descent** là một thuật toán tối ưu phổ biến trong học máy được sử dụng để giảm thiểu hàm chi phí bằng phương pháp lặp đi lặp lại các tham số của mô hình.

- **Gradient** là đạo hàm và **Descent** là đi ngược.

- Hướng tiếp cận của **Gradient Descent** là sẽ xuất phát từ một điểm mà chúng ta cho là *gần* với nghiệm của bài toán, sau đó tiến dần đến điểm cần tìm, tức đến khi gradient gần với 0.
- Dưới đây là các hình ảnh về việc tối ưu tham số của thuật toán GD:

![](/Gradient_Descent/GD_one_variable.png)

![](/Gradient_Descent/GD_var_variables.png)

## Công thức cập nhật
$$
    \mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta \nabla_{\mathbf{w}} J(\mathbf{w}_{t})
$$
Trong đó:
- $\nabla_{\mathbf{w}} J(\mathbf{w}_{t})$ là đạo hàm riêng của hàm cost
- $\eta$ là learning rate, đây là đại lương dương thể hiện tốc độ học của thuật toán.
- Dấu - thể hiện rằng chúng ta sẽ đi ngược dấu đạo hàm.

## Thử nghiệm
### Điểm khởi tạo
- Việc lựa chọn điểm khởi tạo ban đầu cũng sẽ ảnh hưởng đến tốc độ học của mô hình.

![](/Gradient_Descent/1dimg_5_0.1_-5.gif)

![](/Gradient_Descent/1dimg_5_0.1_5.gif)

### Learning rate
- Việc chúng ta lựa chọn learning rate (hay tốc độ học) sẽ ảnh hưởng đến tốc độ hội tụ của thuật toán.
- Có hai trường hợp có thể xảy ra đó là learning rate quá lớn hoặc quá nhỏ.
- Nếu như learning rate quá nhỏ sẽ khiến thời gian hội tụ chậm khi chỉ tiến từng bước nhỏ đến điểm hội tụ.
- Nếu như learning rate quá lớn, tốc độ hội tụ sẽ nhanh hơn nhưng sẽ khiến mô hình khó hội tụ tại global minimum.
- Chúng ta cần thử nghiệm nhiều lần để thu được learning rate phù hợp nhất cho thuật toán.

![](/Gradient_Descent/1dimg_5_0.01_-5.gif)

![](/Gradient_Descent/1dimg_5_0.5_-5.gif)

## Large Dataset và Online Learning

- **Online Learning** là hình thức học khi mà bộ cơ sở dữ liệu được cập nhật liên tục, khiến mô hình phải thay đổi liên tục để phù hợp với bộ dữ liệu.

![](/Gradient_Descent/online_learning.png)

- Khi mà bộ dữ liệu lớn, việc sử dụng toàn bộ dữ liệu cho mỗi lần cập nhật **w** trong quá trình huấn luyện sẽ gây ảnh hưởng đến thời gian huấn luyện và tốn kém về mặt chi phí tính toán. Điều này sẽ không phù hợp với các bài toán *Online Learning*.

- Chúng ta có thể giải quyết vấn đề này bằng cách sử dụng một phần của bộ dữ liệu trong mỗi lần cập nhật, qua cách sử dụng khái niệm là **batch size**, và với các kiểu chọn bacth size khác nhau chúng ta lại có các biến thể của **Gradient Descent**.

## Biến thể của Gradient Descent
![](/Gradient_Descent/variant_gradient_descent.png)
Trước tiên, chúng ta cần đi qua các khái niệm được sử dụng trong phần này:

- Trong **Machine Learning**, **Parameter** là tham số tối ưu của mô hình sau khi đã trải qua quá trình huấn luyện. **Hyperparameter** hay siêu tham số là các tham số được chúng ta lựa chọn sau những lần huấn luyện thử nghiệm.

- **Batch size** là số điểm dữ liệu được đưa vào để cập nhật trọng số cho mô hình. Đây chính là một siêu tham số (hyperparameter) mới mà chúng ta phải chọn.

- **len(dataset)** là số lượng điểm dữ liệu của toàn bộ dữ liệu.
- **Epoch** là khái niệm đại diện cho một lần huấn luyện hết lượng dữ liệu đầu vào.

### Batch Gradient Descent
- Biến thể này sẽ sử dụng toàn bộ dữ liệu để thực hiện cập nhật trọng số, hay **batch size = len(dataset)**.

- Biến thể này sẽ lấy trung bình các đạo hàm của tất cả dữ liệu huấn luyện để có thể cập nhật trọng số, vì vậy mỗi epoch chúng ta sẽ chỉ có một lần cập nhật trọng số.

### Stochastic Gradient Descent
- Biến thể này sử dụng batch size có giá trị bằng 1, tức là sẽ lấy đạo hàm của một điểm dữ liệu cho một lần cập nhật trọng số.

- Với việc cập nhật trọng số nhiều lần trong 1 epoch như thế này sẽ cho đồ thị cập nhật của biến thể có nhiều sự dao động, và không được mượt như Batch Gradient Descent.

- Giải thích cho việc dao động mạnh của SGD: Mỗi điểm đang cố gắng đẩy các tham số theo hướng có lợi nhất cho nó. Cập nhật các tham số có lợi cho một điểm sẽ có thể gây hại cho các điểm khác. Chúng ta có thể giảm các dao động bằng cách tăng dần batch size.

- Khi batch size tăng dần:
    - Thuật toán thực hiện cập nhật gradient ổn định.
    - Thuật toán mất nhiều tài nguyên hơn cho mỗi trainning step.
    - Dễ bị overfit hơn.

### Mini-Batch Gradient Descent
- Biến thể này sử dụng batch size nằm trong khoảng từ 1 đến **len(dataset)**.

- Biến thể này có đồ thị học mượt hơn so với SGD, nhưng lại biến động hơn so với BGD.

- Cách lựa chọn Batch Size:
    - Chúng ta cần đánh đổi trade off với việc lựa chọn batch size, việc lựa chọn này sẽ ảnh hưởng đến độ chính xác, thời gian huấn luyện và tài nguyên tính toán. (Tham khảo thêm: [Batch Size Tradeoff](https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU))
    - Batch size càng lớn sẽ tốn ít thời gian huấn luyện, nhưng độ chính xác sẽ thấp hơn.
    - Chúng ta có thể lựa chọn batch size qua các lần thử nghiệm, và lựa chọn batch size tối ưu nhất.
    - Nên sử dụng batch size = $2^x$, điều này sẽ phù hợp hơn với phần cứng (Tham khảo thêm:[Number of processors in GPU](https://superuser.com/questions/928460/confused-about-gpu-having-hundreds-of-processors-inside-it))
    - Nên bắt đầu với batch size mặc định là 32 và sau đó thử các giá trị khác nếu không hài lòng với giá trị mặc định.
    - Nên bắt đầu với batch size nhỏ trước.
    - Có một mối tương quan giữa batch size và learning rate. Khi learning rate cao, batch size lớn hơn cho kết quả tốt hơn và ngược lại. (Đọc thêm về mối quan hệ giữa learning rate và batch size: [Relation between learning rate and batch size](https://www.baeldung.com/cs/learning-rate-batch-size))
