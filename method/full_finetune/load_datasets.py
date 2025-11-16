import os
os.environ['TMPDIR'] = '/mnt/lv3/chenkaizhe/tmp'
import gc
import random
import librosa
import numpy as np

from datasets import (load_dataset, 
                      concatenate_datasets,
                      IterableDatasetDict,
                      DatasetDict, 
                      Dataset,
                      Audio,
)

from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    TimeStretch,
)

from torchlibrosa.augmentation import SpecAugmentation

def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def spec_augment(features):
    def freq_mask(spec, F=10):
        num_mels = spec.shape[0]
        f = np.random.randint(0, F)  # 掩码长度
        f0 = np.random.randint(0, num_mels - f) # 起点
        spec[f0:f0+f, :] = 0
        return spec

    def time_mask(spec, T=50):
        num_frames = spec.shape[1]
        t = np.random.randint(0, T)  # 掩码长度
        t0 = np.random.randint(0, num_frames - t) # 起点
        spec[:, t0:t0+t] = 0
        return spec
        
    # 应用增强
    features = freq_mask(features)
    features = time_mask(features)
    return features

def create_dataset(
        dataset_dir, ds_keys, audio_paths, transcription_texts,
        sampling_rate, streaming, cache_dir, use_valid_to_train, test_only, languages, json_output_dir
):
    if streaming:
        ds = IterableDatasetDict()
    else:
        ds = DatasetDict()

    for key in ds_keys:
        dataset_dict = {
            "audio": audio_paths[key], "sentence": transcription_texts[key], "language":languages[key]}
        ds_tmp = Dataset.from_dict(dataset_dict)


        os.makedirs(json_output_dir, exist_ok=True)
        
        # 2. 构建新的、唯一的文件路径，避免不同数据集间冲突
        lang_abbr = languages[key][0] if languages[key] else "unknown"
        json_path = os.path.join(json_output_dir, f"{lang_abbr}_{key}.json")
        
        # 3. 写入并读取
        if not os.path.exists(json_path):
            print(f"Writing intermediate json to: {json_path}")
            ds_tmp.to_json(json_path, index=False)
        
        # 读取json文件中信息
        ds[key] = load_dataset("json", data_files=json_path, split='train',
                               features=ds_tmp.features,
                               streaming=streaming,
                               cache_dir=cache_dir,
                               )

    del ds_tmp
    gc.collect()

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["dev"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # add column

    return ds


def load_common_voice(dataset_root, language_abbr="ms_my", sampling_rate=16000, streaming=True, cache_dir="/mnt/lv3/chenkaizhe/.cache/huggingface/datasets", use_valid_to_train=True, test_only=False,  replay_ratio=0.0, json_output_dir="/mnt/lv3/renziang/json_fleurs"):
    
    # dataset_dir = dataset_root +  "CommonVoice/" + language_abbr + "/"
    dataset_dir = os.path.join(dataset_root, "CommonVoice", language_abbr)
    # print("loading dataset dir:", dataset_dir)
    
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]
    
    audio_paths, transcription_texts, languages = {}, {}, {}
    for key in ds_keys:
        # filelist = dataset_dir + f"{key}.tsv"
        filelist = os.path.join(dataset_dir, f"{key}.tsv")
        # print(filelist)
        filepaths_and_text = load_filepaths_and_text(filelist,split='\t')
        filepaths_and_text[0].append("transcription")
        
        # 存入相关信息
        audio_paths[key], transcription_texts[key], languages[key] = [], [], []
        
        # 经验回放，抽取一部分数据集
        if replay_ratio > 0:
            indices = range(len(filepaths_and_text))
            sample = random.sample(indices, round(len(filepaths_and_text) * replay_ratio))
            # print("取样大小: ", len(sample),"/", len(filepaths_and_text),language_abbr)
            # print(sample)
            # print(filepaths_and_text)
            for k in sample:
                if k == 0: 
                    continue
                # audio_path = dataset_dir + "clips/" + filepaths_and_text[k][1]
                audio_path = os.path.join(dataset_dir, "clips", filepaths_and_text[k][1])
                audio_paths[key].append(audio_path)
                if language_abbr == 'zh-CN':
                    transcript = filepaths_and_text[k][2].replace("«","").replace("»","")
                else:
                    transcript = filepaths_and_text[k][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
        else:
            # 如数据集过多可以downsample
            for i in range(1, len(filepaths_and_text)):
                audio_path = os.path.join(dataset_dir, "clips", filepaths_and_text[i][1])
                audio_paths[key].append(audio_path)
                # print(filepaths_and_text)
                if language_abbr == 'zh-CN':
                    transcript = filepaths_and_text[i][2].replace("«","").replace("»","")
                else:
                    transcript = filepaths_and_text[i][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
             
    ds = create_dataset(
    dataset_dir=dataset_dir, ds_keys=ds_keys, audio_paths=audio_paths, transcription_texts=transcription_texts,
    sampling_rate=sampling_rate, streaming=streaming, cache_dir=cache_dir,
    use_valid_to_train=use_valid_to_train, test_only=test_only, languages=languages, json_output_dir=json_output_dir
    )
        
    return ds    

def load_fleurs(dataset_root, language_abbr="ms_my", sampling_rate=16000, streaming=True, cache_dir="/mnt/lv3/chenkaizhe/.cache/huggingface/datasets", use_valid_to_train=True, test_only=False, replay_ratio=0.0, json_output_dir="/mnt/lv3/renziang/json_fleurs"):
    
    # dataset_dir = dataset_root + language_abbr + "/"
    #import pdb; pdb.set_trace()
    dataset_dir = os.path.join(dataset_root, language_abbr)
    # print("loading dataset dir:", dataset_dir)
    
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]
    
    audio_paths, transcription_texts, languages = {}, {}, {}
    for key in ds_keys:
        # filelist = dataset_dir + f"{key}.tsv"
        filelist = os.path.join(dataset_dir, f"{key}.tsv")
        # print(filelist)
        filepaths_and_text = load_filepaths_and_text(filelist,split='\t')
        filepaths_and_text[0].append("transcription")
        # print(filepaths_and_text[0][1])
        
        # 存入相关信息
        audio_paths[key], transcription_texts[key], languages[key] = [], [], []
        # 经验回放，抽取一部分数据集
        if replay_ratio > 0:
            indices = range(len(filepaths_and_text))
            sample = random.sample(indices, round(len(filepaths_and_text) * replay_ratio))
            # print("取样大小: ", len(sample),"/", len(filepaths_and_text),language_abbr)
            # print(sample)
            for k in sample:
                # audio_path = dataset_dir + "audio/" + key + "/" + filepaths_and_text[k][1]
                audio_path = os.path.join(dataset_dir, "audio", key, filepaths_and_text[k][1])
                audio_paths[key].append(audio_path)
                transcript = filepaths_and_text[k][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
        else:
            for i in range(1, len(filepaths_and_text)):
                # audio_path = dataset_dir + "audio/" + key + "/" + filepaths_and_text[i][1]
                audio_path = os.path.join(dataset_dir, "audio", key, filepaths_and_text[i][1])
                audio_paths[key].append(audio_path)
                transcript = filepaths_and_text[i][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
     
    # 创建数据集         
    ds = create_dataset(
    dataset_dir=dataset_dir, ds_keys=ds_keys, audio_paths=audio_paths, transcription_texts=transcription_texts,
    sampling_rate=sampling_rate, streaming=streaming, cache_dir=cache_dir,
    use_valid_to_train=use_valid_to_train, test_only=test_only, languages=languages, json_output_dir=json_output_dir
    )
    print(f"Keys available in ds: {ds.keys()}") # 添加这一行来调试
    # print("ds_train: ", len(ds["train"]))
    return ds    
        
# root /mnt/g2/chenkaizhe/whisper_finetune-master/examples/asr/datasets_asr/

def load_process_datasets(
    datasets_settings, 
    processor, 
    dataset_root,
    json_output_dir,
    streaming=True, 
    cache_dir=None, 
    test_only=False, 
    num_test_samples=1000,
    sampling_rate=16000, 
    max_input_length=30.0, 
    num_proc=1, 
    buffer_size=500, 
    seed=42, 
    language_abbr="id_id", 
    replay_ratio=0.0, 
    augment_data = 0):
    
    # sampling_rate = processor.feature_extractor.sampling_rate
    # all_datasets = {"train": [], "test": []}

    if not cache_dir:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))

    # 函数1: 只处理音频特征，这是无状态的，可以安全地并行
    def process_audio_features(batch):
        audio = batch["audio"]
        # 使用你原来的数据增强逻辑
        if augment_data == 2:
            input_features = processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["input_features"] = spec_augment(input_features)
        else:
            batch["input_features"] = processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return batch

    # 函数2: 只处理文本标签，这是有状态的，但速度很快，我们用单进程处理
    # 注意：这个函数现在需要在所有数据合并后调用
    def process_text_labels(batch):
        language_name = {
            "id_id": "indonesian", "ms_my": "malay", "fil_ph": "tagalog",
            "en_us": "english", "zh_cn": "chinese", "th_th": "thai",
            "vi_vn": "vietnamese", "jv_id": "javanese", "mi_nz": "maori",
            "zh-CN": "chinese", "de_de": "german", "es_es": "spanish",
            "fr_fr": "french", "it_it": "italian", "ja_jp": "japanese",
            "ko_kr": "korean", "pt_pt": "portuguese","cmn_hans_cn": "chinese",
        }
        # 使用 with 上下文管理器，这是处理多语言分词的安全方式
        # 它能确保 tokenizer 的状态只在当前调用中改变，不会影响其他进程
        lang = language_name[batch["language"]]
        with processor.as_target_processor():
            processor.tokenizer.set_prefix_tokens(language=lang, task="transcribe")
            batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    
    
    def augment_dataset(batch):
        # load and (possibly) resample audio data to 16kHz
        sample = batch["audio"]
        
        # 数据增强操作
        augmentation = Compose(
                [
                    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2,
                                leave_length_unchanged=False),
                    Gain(-6, 6, p=0.1),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                    OneOf(
                        [
                            AddGaussianNoise(min_amplitude=0.005,
                                             max_amplitude=0.015, p=1.0),
                        ],
                        p=0.2,
                    ),
                ]
            )
        
        augmented_waveform = augmentation(
            sample["array"], sample_rate=sample["sampling_rate"])

        batch["audio"]["array"] = augmented_waveform

        return batch
    
    # --- 步骤 2: 重新组织数据加载和处理流程 ---

    train_list, test_list = [], []
    for name, kwargs in datasets_settings:
        print(f"Loading dataset: {name} with args {kwargs}")
        ds_tmp = None
        if name == "fleurs":
            if "language_abbr" not in kwargs:
                raise ValueError("language_abbr must be specified for fleurs dataset")
            ds_tmp = load_fleurs(
                dataset_root, 
                sampling_rate=sampling_rate,
                streaming=streaming, 
                cache_dir=cache_dir, 
                test_only=test_only, 
                replay_ratio=replay_ratio,
                json_output_dir=json_output_dir,
                **kwargs)
            print(f"fleurs-{kwargs}: ", next(iter(ds_tmp["test"])))

        elif name == "common_voice":
            if "language_abbr" not in kwargs:
                raise ValueError("language_abbr must be specified for common_voice_local dataset")
            ds_tmp = load_common_voice(
                dataset_root, 
                sampling_rate=sampling_rate,
                streaming=streaming, 
                cache_dir=cache_dir, 
                test_only=test_only, 
                replay_ratio=replay_ratio,
                json_output_dir=json_output_dir,
                **kwargs)
            print(f"common_voice-{kwargs}: ", next(iter(ds_tmp["test"])))

        if ds_tmp is not None:
            test_list.append(ds_tmp["test"])
            if not test_only:
                train_list.append(ds_tmp["train"])
                # 数据增强
                if augment_data == 1: # 注意：你原来的代码 augment_data==2 似乎是频谱增强，这里只处理波形增强
                    # 修复硬编码的并行数！
                    augmented_raw_training_dataset = ds_tmp["train"].map(
                        augment_dataset, num_proc=num_proc, desc="augment train dataset")
                    train_list.append(augmented_raw_training_dataset)

    # 合并所有数据集
    ds = DatasetDict()
    ds["test"] = concatenate_datasets(test_list)
    if not test_only:
        ds["train"] = concatenate_datasets(train_list)

    # --- 步骤 3: 分步执行 map 操作 ---

    print("🚀 Step 1/3: Processing audio features...")
    # 首先，只处理音频，这个过程可以安全地并行
    ds = ds.map(
        process_audio_features, 
        num_proc=num_proc, 
        writer_batch_size=100  # <--- 加入這一行！
    )

    print("🚀 Step 2/3: Processing text labels...")
    # 然后，处理文本，这个过程用单进程来避免死锁
    ds = ds.map(process_text_labels, num_proc=1) # 使用 num_proc=1 保证安全

    print("🚀 Step 3/3: Filtering and formatting...")
    # 最后，进行过滤和格式化
    def is_audio_in_length_range(length):
        return length < max_input_length

    ds = ds.filter(is_audio_in_length_range, input_columns=["input_length"], num_proc=num_proc)

    if not streaming:
        ds = ds.shuffle(seed)
        num_test_samples = min(num_test_samples, ds["test"].num_rows)
        ds["test"] = ds["test"].select(range(num_test_samples))
    
    # 移除不再需要的原始列
    ds = ds.remove_columns(["audio", "sentence", "input_length"])
    ds = ds.with_format("torch")

    print("✅ Final dataset ready!")
    print(ds)
    return ds



if __name__ == "__main__":
    from transformers import WhisperProcessor, WhisperTokenizer

    # Model setups
    model_name_or_path = "Oblivion208/whisper-tiny-cantonese"
    task = "transcribe"
    language = "zh"
    # Dataset setups
    datasets_settings = [
        ["mdcc", {}],
        ["common_voice", {"language_abbr": "zh-HK"}],
        ["aishell_1", {}],
        ["thchs_30", {}],
        ["magicdata", {}],
    ]
    max_input_length = 30.0
    num_test_samples = 1000

    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, task=task)
    ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=max_input_length,
        num_test_samples=num_test_samples,
        test_only=True,
        streaming=False,
        num_proc=4,
    )
    print(ds)
    print("test", ds["test"][:10]["input_length"])
    # print("train", ds["train"][:10]["input_length"])
    # print("test sample: ", next(iter(ds["test"])))
