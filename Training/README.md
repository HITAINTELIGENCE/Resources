# Training and Optimization Concept

## Underfiting và Overfiting
- **Underfitting (high bias)** xảy ra khi mô hình quá đơn giản và không nắm bắt các feature cơ bản trong data, dẫn đến hiệu suất kém trên các tập train và test.
- **Overfiting (high variance)** xảy ra khi mô hình quá khớp với training data, dẫn đến hiệu suất trên dữ liệu không được huấn luyện trên tập test thấp, hay nói cách khác, độ chính xác đối với tập test sẽ thấp hơn tập huấn luyện.

- Overfiting xảy ra trong trường hợp mô hình quá phức tạp để mô phỏng dữ liệu huấn luyện, hay nói cách khác chúng ta đang có ít dữ liệu huấn luyện mà mô hình thì quá phức tạp.

![](/Training/Image/fitting.png)


## Đánh giá mô hình 
Có hai đại lượng có thể giúp chúng ta đánh giá xem mô hình này có thực sự tốt hay không, dựa trên training data và test data chính là train error và test error
- Train Error

$$
train error = \frac{1}{N_{\text{train}}} \sum_{\text{training set}} \| y - \hat{y} \|_2^p
$$

- Validation Error

$$
test error = \frac{1}{N_{\text{test}}} \sum_{\text{test set}} \| y - \hat{y} \|_2^p
$$

- Lưu ý: Chúng ta cần chia trung bình vì có sự chênh lệch về số lượng phần tử giữa hai tập dữ liệu


Xét các trường hợp sau đây:
- Nếu train error thấp, test error cao thì mô hình rơi vào hiện tượng overfiting
- Nếu train error cao, test error cao thì mô hình rơi vào hiện tượng underfiting



## Các phương pháp tránh overfiting và underfiting

### Underfiting
- Cải thiện mô hình: Chúng ta có thể cải thiện mô hình bằng cách thêm các polynomial feature

![](/Training/Image/imp_underfiting.png
)


### Overfiting

- Phương pháp đầu tiên chính là sử dụng **validation**, ở đây chính là trích một tập đánh giá từ tập dữ liệu huấn luyện, và chúng ta lại có khái niệm về validattion error

- Để biết mô hình có bị underfiting hay overfiting không chúng ta cần phải đánh giá mô hình dựa trên data dùng để đánh giá. Dataset thường được chia thành 3 phần:
    - Training data: dùng để huấn luyện mô hình
    - Validattion data: dùng để đánh giá model trong quá trình huấn luyện
    - Test data: dùng để đánh giá model sau khi huấn luyện xong

- Nếu mà mô hình có train error và validation error thấp, thì đó chính là mô hình tốt.

![](/Training/Image/linreg_val.png)

- Một phương pháp khác cải tiến khi tập validation quá nhỏ là **cross validation**, với phương pháp này có rất nhiều tập validation nhỏ nhưng lại được đánh giá trên nhiều tập khác nhau

![](/Training/Image/cross-validation.png)

- **Regulization** chính là phương pháp tiếp theo được sử dụng trong học máy để giảm overfiting, nó sẽ thay đổi mô hình và vẫn giữ được tính tổng quát dữ liệu
    - Thêm số hạng vào hàm mất mát
    $$

    J_{\text{reg}}(\mathbf{w}) = J(\mathbf{w}) + \lambda R(\mathbf{w})

    $$

    Trong đó:
    - $R(\mathbf{w})$ số hạng *regularization* 
    - $\lambda$  lớn hơn hoặc bằng 0, quyết định mức độ ảnh hưởng của $R(\mathbf{w})$.

    ![](/Training/Image/regulization.png)

Một trong những lý do dẫn điến việc Overfiting là do có quá nhiều feature nên mô hình có thể phù hợp rất tốt với tập train, nhưng không thể khái quát hóa thành các example mới. Thêm số hạng vào hàm mất mát giúp giảm các giá trị weight đến mô hình từ đó cho mô hình mang tính tổng quát hơn.

Xét $l_2$-**regularization**

$$
R(\mathbf{w}) = \|\mathbf{w}\|_2^2
$$

Khi đạo hàm loss function ta được:

$$
\frac{\partial J_{\text{reg}} }{\partial \mathbf{w}} = \frac{\partial J}{\partial \mathbf{w}} + \lambda \mathbf{w}
$$

Từ hàm loss trên có thể thấy giá trị loss công thêm $\lambda \mathbf{w}$ → giá trị hàm loss lớn hơn → weight cập nhật sẽ nhỏ hơn → mức ảnh hưởng của feature trở lên nhỏ hơn → mô hình mang tính tổng quát hơn.

- **Thu thập thêm dữ liệu**

![](/Training/Image/collect-data.png)

- **Early Stopping**, hay dừng vòng lặp sớm, bằng cách dựa trên train error và validation error bằng cách:
    - Nếu mà validation error có chiều hướng tăng hoặc nhỏ hơn ngưỡng nào đó, thì sẽ dừng vòng lặp
    - So sánh với giá trị của model trước đó, ta sẽ lấy train error nhỏ nhất

![](/Training/Image/early-stopping.png)

- **Dropout** được sử dụng trong Neural Network, đây là phương pháp tắt ngẫu nhiên các unit trong mạng. Tắt ở đây chính là cho các unit này có giá trị bằng 0 và tính toán feedforward và backpropagation bình thường trong khi training. Việc này không giúp những lượng tính toán giảm đi mà còn làm giảm overfiting.

![](/Training/Image/dropout.png)


# Unbalancing Data

- Mất cân bằng dữ liệu là hiện tượng phổ biến của bài toán phân loại nhị phân (binary classification), xuất hiện khi dữ liệu của bài toán không ở trạng thái cân bằng về số lượng ở mỗi class.

- Đối với bài toán mà có dữ liệu không được cân bằng, thì metric đánh giá là accuracy sẽ không còn được chính xác nữa.

## Các phương pháp giaỉ quyết
- Thay đổi phương pháp đánh giá sang precision, recall, f1-score, ...

![](/Training/Image/crossTable.png)

- Trong bảng trên, chúng ta cần lưu ý về các chỉ số:
    - Precision: Mức độ dự báo chính xác trong những trường hợp được dự báo là Positive.
    - Recall: Mức độ dự báo chuẩn xác những trường hợp là Positive trong những trường hợp thực tế là Positive.
    - F1-Score: Trung bình điều hòa giữa Precision và Recall. Đây là chỉ số thay thế lý tưởng cho accuracy khi mô hình có tỷ lệ mất cân bằng mẫu cao.

- Nếu mô hình có các chỉ số kia càng cao, thì mô hình sẽ có chất lượng càng tốt.


