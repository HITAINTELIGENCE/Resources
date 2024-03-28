1. Các bạn hoàn thành các yêu cầu trong file .ipynb nhé
2. Dưới đây là cơ sở lý thuyết của Linear Regression
3. Các bạn có thể xem bản khác trên Notion: [Notion](https://tremendous-chili-4f0.notion.site/Linear-Regression-0467b25e5cc6466494baddc7bdb67789)
# Linear Regression

- Ý nghĩa của tên Linear Regression
    
    - **Regression**: là một tập hớp các xử lý thống kê để đánh giá mối quan hệ giữa các các **dependent variable** (biến phụ thuộc) với một hoặc nhiều **independent variable** (biến độc lập).
    
    - **Linear**: là đường thẳng, mặt phẳng, siêu phẳng.
    

- **Linear Regression** là thuật toán cố gắng tìm một phường trình: đường thẳng, mặt phẳng, siêu phẳng,… **fit** với bộ data traning (tìm giá trị $\mathbf{w}$ và $b$) thể hiện **linear relationship** của các feature vs target. Với mục đích để dự đoán một biến dựa trên giá trị của biến khác. Giá trị muốn predict được gọi là **dependent variale**, giá trị được sử dụng để predict được gọi là **independent variable**.

- Công thức tổng quát cho linear regression:

$$
\mathbf{y} = \mathbf{w}X + b
$$

![](/Linear_Regression/linearregression.png)

## Cost function

- Có nhiều hàm loss cho linear regression, nhưng tất cả đều hướng đến mục đích đưa ra được sự chênh lệch giữa giá trị predict và giá trị thực tế, để giúp cải thiện mô hình. Các hàm loss phổ biến: [loss function](https://www.statlect.com/glossary/loss-function)

- Sự khác biệt giữa các hàm loss chủ yếu đến từ việc đánh trọng số khác nhau đối với các giá trị chênh lệch giữa hai giá trị predict và thực tế, việc này có thể cải thiện mô hình trong các hoàn cảnh khác nhau. Xem thêm về sự khác biệt giữa 3 hàm loss phổ biến: [Quadratic & Absolute & Huber loss](https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3)

### **MSE cost function**

$$
J(\mathbf{w}) = \frac{1}{2m}\sum_{i=1}^N (\mathbf{w}\mathbf{x_i} + b - y_i)^2
$$


![](/Linear_Regression/lossandcost.png)

- **Chú ý**
    
    Loss function $(\mathcal{L})$ và Cost function $(J)$ có chút khác biệt: loss function là giá trị chênh lệch giữa hai giá trị predict và thực tế, trong khi cost function là tổng trung bình các giá trị loss.
    
    Lý do của việc lấy trung bình (chia cho $m$) là để tránh việc giá trị của hàm cost quá lớn gây ảnh hưởng đến việc học của mô hình.
    
    Chia thêm cho 2 với mục đích giúp cho việc tính gradient dễ dàng hơn: $(\frac{1}{2}x^2)' = x$ 
    

## Figure out $\mathbf{w}$ and $b$

Có hai cách chính để tìm ra giá trị $\mathbf{w}$ và $b$:

- Normal gradient: Tìm nhiệm khi đạo hàm bằng 0 ([machinelearningcoban](https://machinelearningcoban.com/2016/12/28/linearregression/), [Newton’s method](https://machinelearningcoban.com/2017/01/16/gradientdescent2/#-mot-phuong-phap-toi-uu-don-gian-khac-newtons-method)).
- [Gradient descent](https://www.notion.so/Gradient-descent-15d2f81a4b6f43ae8faf5e18bd371576?pvs=21).