
## 1. 資料前處理 (raw data pre-processing)

### 01 圖像增強 (image augmentation)

1. 在PyTorch中，有torch.utils的`DataLoader`，可以讓我們創建dataloader的物件。
   並且可以在這個物件中，設定image要怎麼做前處理(例如data augmentation)。
2. 通常使用data augmentation是使用torchvision.transforms的功能。
    - 例如要把圖像做resize.
    - flip, rotation.
    - 還有就是要將pixel value標準化在(0,1)值中間，
      這樣才能feed in 到pre-trained的移轉學習的model裡面。
    - 當然這些標準化的值，還需要變成tensor物件才行。
3. 記得模型在訓練和測試時，要放進入的圖片，所需要做的前處理不同，
   在測試時，因為要了解模型是不是真的能夠準確的判斷圖片中的feature，
   因此在testing時，就不需要做那一些flip或是rotation的動作。

### 02 文字的篩選 (caption pre-processing)

1. 在決定哪些字要收入vocabulary集時，可以先決定vocab threshold，
   也就是哪個字，在文本裡至少出現多少次，才要收錄進去，要不然就以<unkown>
   來表示是'不明字'。
   - 在其它的NLP案例的應用中，也會直接就將罕見字刪除，
   還有就是哪個字，如果出現太多次(常見字)，也要把它刪除掉，例如stop words等。
2. 記得每一串caption，都還需要有<start>和<end>的token，
   讓模型可以辨識什麼時候是字串的開始，什麼時候是字串的結束。
3. 接著，就是要將這些字，變成tensor，這樣放入PyTorch中。
4. 在文字bag of words的前處理中，還創造了index對應每個word的dictionary，
   叫做`word2idx`。
   
## 2. 建立CNN Encoder

### 01 要放入的input值

1. 如同在[TORCHVISION.MODELS](https://pytorch.org/docs/master/torchvision/models.html)
   裡面所說的：
   
   > All pre-trained models expect input images normalized in the same way, i.e. 
   mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least `224`. 
   The images have to be loaded in to a range of \[0, 1\] and then normalized using mean = \[0.485, 0.456, 0.406\] 
   and std = \[0.229, 0.224, 0.225\].
   
2. 因此原本的圖片，就要轉換成size (3, 224, 224)作為input.
3. 之後Encoder的output，就會是我們設定的`embed size`.


## 3. 建立LSTM Decoder

![decoder structure](images/decoder.png)

### 01 要放入的input值

1. 如同上圖所示，在第一個LSTM的input，就是由CNN encoder最後所產出的embeded vector.
2. 之後的每一個LSTM的output，就是target caption的前一個預測的字。
3. 但是這個預測的字，還需要先經過一個word embed的過程，讓字變成是統一dimension的embeded vector.
    - 也因此，我們可以看到LSTM模型中，還有一層layer是`nn.Embedding`。它的功用就是一個lookup table.

### 02 LSTM的output值

1. 之後就是output一個`vocab_size`的vector，每個element都是一個字的預測機率，
   之後挑出最大的預測機率的值。-> 採用簡單的最大機率的方式(也就是模型裡面的`sample()` method)
2. 還有另外一個方式，是使用**beam search**，這也是在`beam_search()` method裡面使用的。
    - 在`beam_search()`裡面，還有透過幾個intermediary function來達成這個功能。
        1. 記得，beam_search是只有在testing時，才會使用的。-> training時，是不使用beam_search的。
    - 一個是`forward_features()`：用途在feed in the first embeded vector from CNN Encoder.
        1. 記得函數裡面，創造了hidden_state，讓這個hidden_state可以一直傳遞下去。
    - 一個是`forward_word()`：之後再針對每個每個top K words，視作是batch_size為K，
      進行每個word下面預測的字串。
      
        ```python
        scores = scores.squeeze(1) # (k, vocab_size)
        scores = seq_scores.expand_as(scores) + scores
        ```
    
        1. 可以看到，有一個`seq_scores.expand_as()`的動作，
           就是要讓原本top K的機率值，再加上每一個vocab_size裡，
           每個字的機率值。之後用`beam_search()`來篩選出top K
           (`top_scores, top_ids = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)`)
           的字。
        2. 不要忘記了，beam_search裡面，還有一個功能是設定caption的長度，有限制(預設最大是20個字)
        3. 在beam_search，每一次的篩選，都是先選出前面K的字，append上去之前的字串後，
           再篩選出到目前為止，每個字的機率(經過log轉換後)加總起來，值最大的字串(並且只保留K個)。

### 03 LSTM layer的數量

1. 因為 image captioning 容易出現over-fitting的狀況，因此根據
   [Show and Tell: A Neural Image Caption Generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
   這一篇文章，它建議是只要使用一層LSTM layer就好。


## 4. 後記

### 01 沒有使用attentive model

1. 在這裡，我沒有使用attentive model，因為attentive model需要將CNN Encoder的資訊，
   直接放入到LSTM的hidden state裡面，這樣必需使用`nn.LSTMCell`的物件，但是我發現，
   使用這個物件所訓練出來的模型，怪怪的，就像是沒有訓練過的模型一樣。
    - 在網路上，也有人提到，使用nn.LSTMCell的話，它在cuda上頭是沒有經過優化的，所以不建議使用。 
2. 因此我改使用`nn.LSTM`來直接建構一整個LSTM模型。

### 02 Hyper-parameter的設定&Optimizer

1. batch_size = 128 -> 可以增加訓練的速度(但同時比較不容易找到最佳解，
   因為iteration的次數較少)
2. vocab_threshold = 6 -> 每個字至少要出現6次以上，才收錄在字典裡。
3. embeded_size and hidden_size 均設定為512 
   -> following what's used in **show and tell image caption** paper.
4. 針對Adam optimizer的設定，我根據
   [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
   的設定。
    1. 它的default value是 (beta1, beta2) = (0.9, 0.999)，通常使用default value，就可以得到很好的結果了。
    2. **重點**：使用Adam的優點有哪些？
        - Straightforward to implement.
        - Computationally efficient.
        - Little memory requirements.
        - Invariant to diagonal rescale of the gradients.
        - Well suited for problems that are large in terms of data and/or parameters.
        - Appropriate for non-stationary objectives.
        - Appropriate for problems with `very noisy/or sparse gradients`.
        - Hyper-parameters have intuitive interpretation and typically require little tuning.
    3. Adam 基本上就是AdaGrad和RMSProp兩種optimizer優點的結合體。
        - AdaGrad 擅於使用在sparse的gradient.
        - RMSProp 則擅於使用在noisy或是non-stationary的gradient
          (因為它會平均前面好幾期的梯度).

### 03 data_loader 這個物件的描述

1. 它會使用到PyTorch裡面的兩個功能，一個是data.DataLoader
2. 一個是data.sampler.SubsetRandomSampler
    - 主要目的是把喂入的資料集，用without replacement的方式，
      製造出已經被sampling的indices。
3. 一個是data.sampler.BatchSampler
    - 主要的目的，是將所有的sampling，在用一個batch，一個batch的方式，丟出來給模型。 
4. 接著，在這個DataLoader裡面，會再使用一個coco dataset的物件。
    - 首先，它會先用get_train_indices，來先選出指的定caption長度，
      之後再將和這個長度一樣的captions，全部抓出來，並且sample for the batch size
    - 之後data.DataLoader就會利用這組indices，來叫出相對應的image and caption
      (當然，它後面還是利用到了 \_\_getitem\_\_ 這個方法，這個是利用PyTorch裡面dataset物件的內置方法)
5. 記得在coco dataset裡面，還用到了自己設定的Vocabulary的物件。
    - 這個物件，就是將所有caption裡面的字，都tokenized，之後存入兩個字典中，分別是
        - word2idx
        - idx2word
    - 它借用的方法，就是`add_word()`和`add_captions()`這兩個自訂的方法。
    - 順便可以再複習一下\_\_call\_\_ 和 \_\_init\_\_ 這兩個方法，有什麼不同，請參考[this page](https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call-in-python)  
         
6. 在PyTorch當中，針對自訂DataLoader的使用介紹，可以參考[this page](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)







 
         

 









