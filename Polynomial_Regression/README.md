1. Dưới đây là cơ sở lý thuyết của Polynomial Regression
2. Các bạn có thể xem bản khác trên Notion: [Notion](https://www.notion.so/Polynomial-Regression-2228a27a75194ac9b3f739c22591d1f9)
# Polynomial Regression
## Khái niệm
- Polynomial Regression là một dạng thuật toán đặc biệt của Linear Regression nhưng mối quan hệ giữa các feature và label không phải dạng tuyến tính.
- Polynomial Regression vẫn là dạng thuật toán Regression, nhưng phù hợp để fit với các dạng dữ liệu dạng đường cong.
![](/Polynomial_Regression/polynomial.png)

## Non Linear Relationship Problem
- Đối với các bài toán có mối quan hệ giữa các feature và label không phải tuyến tính, chúng ta không thể biểu diễn bằng đường thẳng.
- Trong toán học chúng ta cũng gặp nhiều trường hợp về các hàm số không có tính tuyến tính như \(\sin(x)\), \(\cos(x)\), \(\log(x)\), ... hay các đa thức bậc cao có chứa các thành phần như \(x^4\), \(x^3\), \(x^2\),..., hãy xem các ví dụ dưới đây

$$ 
        y = log(x) 
$$

![](/Polynomial_Regression/log_x.jpg)
$$
    y = x^3 + 5x^2 + 2x + 5
$$
![](/Polynomial_Regression/polynomial_vis.png)


## Ý tưởng
-  Ý tưởng cơ bản của bài toán này là sử dụng **polynomial feature transformation** để thu được một mô hình fit nhất với dữ liệu.

- **Polynomial Feature trasformation** là kỹ thuật chúng ta mở rộng không gian đặc trưng bằng cách tạo ra các biến độc lập mới thông qua các biến độc lập ban đầu.
    - **VD**: Một bài toán dự đoán giá nhà sử dụng feature bình phương diện tích ngôi nhà, chứ không sử dụng diện tích thực của ngôi nhà đó.
    

- **Lưu ý**: Việc lựa chọn bậc của hàm số là bước rất quan trọng đối với thuật toán này, bậc càng cao sẽ giúp cho mô hình fit hơn dữ liệu được huấn luyện, nhưng sẽ có thể dẫn đến hiện tượng overfiting.

