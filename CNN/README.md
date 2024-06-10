# CNN - Mạng tích chập
## Idea

Một trong những cách tạo ra feature mới là đưa các feature đã biết vào các mô hình như linear regression, logistic regression (xem lại [Feature engineering](https://www.notion.so/Feature-engineering-64ba35f283a0439e8edddb17c0e01c97?pvs=21)),... 

![illustration cho logistic regression unit](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f50df826-5e59-40bf-aeb7-62964a0577de/Untitled.png)

illustration cho logistic regression unit

Khi sử dụng nhiều mô hình để tạo ra nhiều features thì sẽ như thế nào? Xem ví dụ dưới đây:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9d8ed427-f2cb-4cd0-92e4-ebb4910b633c/Untitled.png)

Ta có thể thấy *affordability feature* được tạo bởi *price* và *shipping cost*,… Ta được 3 features mới và tiếp tục sử dụng chúng để predict ra *probability of being a top seller*. 

Ví dụ là như vậy nhưng trên thực tế ta khó có thể biết được những features nào thật sự tốt và cách để tìm ra chúng. Vì vậy ý tưởng là ta sử dụng tất cả các feature đã biết:

 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/be8cd419-56a5-48b9-887a-83c15ba0570c/Untitled.png)

Ta sử dụng tất cả các feature đã biết để tạo ra một feature mới, mô hình sẽ học và chọn ra các features nào cần thiết một cách tự động bằng cách update các trọng số cho mỗi features.

Các thuật toán nhỏ để đưa ra feature mới được gọi là *neuron/unit*, tổng hợp các unit của mỗi tầng được gọi là *layer*, và tổng hợp các layer được gọi là *Neural networks.*

- **Example: : Recognizing Images**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0ab403b5-4c3e-48b9-8a28-f948d33bcb1e/Untitled.png)
    
    Hình trên cho ta thấy dựa vào các pixels của ảnh qua layer đầu tiên có thể biết được các cạnh, qua layer thứ hai có thể biết các bộ phận,… Cứ như vậy càng sau ta lại càng có thêm nhiều thông tin chi tiết hơn về hình ảnh.
    

## Structure

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3305699e-a11c-4d74-9068-c4e21da22d85/Untitled.png)

Một mạng NN có thể chia làm 3 phần chính: 

- Input layer: Các feature đầu vào.
- Hidden layers: Các layers trung gian chịu trách nhiệm extract feature.
- Output layer: Bộ phân loại, gồm các units chứa các thuật toán phân loại như sigmoid, softmax, SVM...

### Hidden layers

Các layers gồm các units có cấu trúc gồm thuật toán Linear Regression và một activation phi tuyến tương tự sư Logistic Regression.

Việc có activation là cần thiết để hàm NN có thể xấp xỉ mọi hàm số bằng các phương trình bậc một đơn giản, nếu không có một hàm phi tuyến thì NN sẽ không khác gì một thuật toán Linear Regression tiêu chuẩn.

![Các đường boudary được tạo thành thừ các đường thẳng ngắn](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41f83b35-89fb-44d3-87ce-9d629467ca5a/Untitled.png)

Các đường boudary được tạo thành thừ các đường thẳng ngắn

Một số activation phổ biến hay gặp: ReLU, Leaky ReLu, sigmoid, tanh…

## Learning

Giống như các thuật toán ML khác, NN cụng học dựa vào Gradient Descent với hàm Loss phù hợp.

Đạo hàm của từng tham số được bằng quy tắc **chain rule** hay đạo hàm hợp, thuật toán này được gọi là **Backpropagation** ([xem thêm](https://machinelearningcoban.com/2017/02/24/mlp/#-backpropagation)).

![Đường màu đen thể hiện forward, màu đỏ thể hiện backward (tính đạo hàm).](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ad7073cb-58a3-4caa-b7e7-1eed7023fd95/Untitled.png)

Đường màu đen thể hiện forward, màu đỏ thể hiện backward (tính đạo hàm).

![Performance của NN](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7bb9c6be-6825-40b5-9de6-5852cdc3448f/Untitled.png)

Performance của NN

# Notice

Việc xử dụng tất các feature không có nghĩa là các kĩ thuật **Feature Engineering** như *feature selection*,… không còn hiệu quả. Sử dụng các feature biết chắc là không cần thiết sẽ gây lãng phí tài nguyên. Việc sử dụng các kĩ thuật **Feature Engineering** sẽ luôn giúp cải thiện **preformace** của mô hình. Lý do có thể đọc [What is Data preprocessing, Why Do We Need That?](https://medium.com/nerd-for-tech/what-is-data-preprocessing-why-we-need-that-2846b8b04bc4)

## Init weights

Việc khởi tạo weight cần được xem xét một các cẩn thận, hãy xem tại sao:

![Độ thị hàm sigmoid và đạo hàm tương ứng](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9cc09210-9388-4230-81e3-21a7eb2291f2/Untitled.png)

Độ thị hàm sigmoid và đạo hàm tương ứng

Ta có thể thấy nếu với giá trị x quá lớn hoặc quá nhỏ sẽ khiến cho đạo hàm của sigmoid sấp xỉ bằng 0, và khi đạo hàm bằng 0 thì ta không thể áp dụng Gradient Descent hiện tượng này gọi là **vanishing gradient**.

Quay trở lại với NN, khi weight khởi tạo quá lớn thì đầu ra sẽ là một số rất lớn và cho đi qua activation như sigmoid thì hiện tượng **vanishing gradient** sẽ xảy ra, một số trường hợp có thể xảy ra **explore gradient** tức đạo hàm quá lớn.

Để giải quyết vấn đề trên cách đơn giản là scale các giá trị khởi tạo của weight. Có thể nhắc đến các phương pháp:

- Xavier: được chứng minh là hiệu quả với sigmoid và tanh
    
    ![inputs và outputs tương ứng số lượng units đầu vào và ra](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc8dcb17-c6ad-4c0e-bf73-fe39556226d4/Untitled.png)
    
    inputs và outputs tương ứng số lượng units đầu vào và ra
    
    https://drive.google.com/file/d/1QKK0x-UcJolloOmCMLFtuN7C6LmhLnTJ/view?usp=drivesdk
    
- He/Kaiming initialization: Được chứng minh hiệu quả với ReLU và các biến thể của nó
    
    ![$n_{in}$ số lượng units đầu vào](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/77283409-ef4a-47cc-91f7-7123269f758c/Untitled.png)
    
    $n_{in}$ số lượng units đầu vào
    
    https://drive.google.com/file/d/1ZF85cAEBOjb0Jo6ZHX22tcLhbGhuOCx4/view?usp=drivesdk
    

### Note

Với Keras, **Xavier initialization** được đặt làm mặc định.

Với Pytorch, **Lecun initiation** với khoảng giá trị từ $[-\frac{1}{\sqrt{n_{in}}}, \frac{1}{\sqrt{n_{in}}}]$ được đặt làm mặc định.

Xem thêm:

[Part 2: Selecting the right weight initialization for your deep neural network. | by Gideon Mendels | Medium](https://medium.com/@gidim/part-2-selecting-the-right-weight-initialization-for-your-deep-neural-network-cc27cf2d5e56)

[Initializing neural networks - deeplearning.ai](https://www.deeplearning.ai/ai-notes/initialization/index.html)