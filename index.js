const MODEL_PATH = 'model_js/model.json';
const MODEL_PATH_2 = 'model_js/New folder/model.json'
const IMAGE_SIZE = 224;
const CLASS_LABELS1 = {
    0: 'ADULTERATED',
    1: 'UNADULTERATED',
}
const CLASS_LABELS2 = {
    0: 'RAW',
    1: 'OVERRIPE',
    2: 'RIPE'
}

const selected_image = document.getElementById('selected-image');
const image_selector = document.getElementById('image-selector');
const status_message = document.getElementById('status');
const predict_result = document.getElementById('prediction-result');
const predict_result_2 = document.getElementById('prediction-result-2');
const prediction_confidence_1 = document.getElementById('prediction-confidence-1');
const prediction_confidence_2 = document.getElementById('prediction-confidence-2');
const nutri = document.getElementById('nutri');

const status = msg => status_message.innerText = msg;

let model;
let model2;
const modelDemo = async () => {
    status('Loading model...');
    // load model
    model = await tf.loadLayersModel(MODEL_PATH);
    model2 = await tf.loadLayersModel(MODEL_PATH_2);

    // warmup
    model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
    model2.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
    status('');

    // predict sample image
    if (selected_image.complete && selected_image.naturalHeight !== 0) {
        predict(selected_image);
    } else {
        selected_image.onload = () => {
            predict_button(selected_image);
        }
    }
};


async function predict(image) {
    // clean text before predicting
    predict_result.innerText = '';
    predict_result_2.innerText = '';

    status('Predicting...');
    const startTime = performance.now();
    const logits1 = tf.tidy(() => {
        const img = tf.browser.fromPixels(image).toFloat();
        const offset = tf.scalar(127.5);
        const normalized = img.sub(offset).div(offset);
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        return model.predict(batched);
    });

    const logits2 = tf.tidy(() => {
        const img = tf.browser.fromPixels(image).toFloat();
        const offset = tf.scalar(127.5);
        const normalized = img.sub(offset).div(offset);
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        return model2.predict(batched);
    });

    showResultOfModel1(logits1);
    showResultOfModel2(logits2);
    
    const totalTime = performance.now() - startTime;
    status(`Done in ${Math.floor(totalTime)}ms.`);
}

async function showResultOfModel1(logits) {
    const message = {
        0: 'ADULTERATED',
        1: 'UNADULTERATED',
    }

    const values = await logits.data();
    console.log(values);

    const class_idx = logits.argMax(1).dataSync();
    const max = 100*Math.max(...values);
    predict_result.innerText = message[class_idx];
    prediction_confidence_1.innerText = max;
    // if (class_idx==0) {
    //     nutri.innerText = 'Not suitable for eating';
    // }else{
    //     nutri= class_idx;
    // }
}

async function showResultOfModel2(logits) {
    const message = {
        0: 'RAW',
        1: 'OVERRIPE',
        2: 'RIPE'
    }
    
    const values = await logits.data();
    console.log(values);

    const class_idx = logits.argMax(1).dataSync();
    const max = 100*Math.max(...values);
    
    predict_result_2.innerText = message[class_idx];
    prediction_confidence_2.innerText = max;
    // if (nutri==1 && class_idx==0) {
    //     nutri.innerText = 'Suitable for diabetics' <br>'Not suitable for people with slow or weak digestive system';
    // }else if (nutri==1 && class_idx==1) {
    //     nutri.innerText = 'Suitable for: people working out in the gym, people with weaker digestive systems, running( good for cardiovascular health)' <br>'Not suitable for: type 2 diabetic patients';
    // }else if (nutri==1 && class_idx==2) {
    //     nutri.innerText = 'Suitable for weight loss when consuming less than 2 bananas as it contains low fat, as a normal snack, gym people' <br> 'Not suitable for people who suffer from asthma, weight loss if consumed more than 2 as it is densely packed with calories';
    // }
    
}

image_selector.addEventListener('change', evt => {
    let files = evt.target.files;

    for (let i = 0, f; f = files[i]; ++i) {
        if (!f.type.match('image.*')) { continue; }

        let reader = new FileReader();
        reader.onload = e => {
            selected_image.src = e.target.result;
            selected_image.width = IMAGE_SIZE;
            selected_image.height = IMAGE_SIZE;
            selected_image.onload = () => predict(selected_image);
        };

        reader.readAsDataURL(f);
        break;
    }
});


modelDemo();
// modelDemo2();
