import os
from train import *
from train_2 import *
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')

# Linear Regression and MLP (Dense)
def test_model(models, input_transform, output_transform, otu_columns, bioma_transfer_test, domain_transfer_test):
    data_bioma_test_transformed = Percentage()(bioma_transfer_test)
    if input_transform is not None:
        input_transform = input_transform()
    if output_transform is not None:
        output_transform = output_transform()
    metrics_results = {}
    metrics = get_experiment_metrics(input_transform, output_transform)[0][3:]
    otus_errors = []
    all_predictions = []
    for cv_models in models:
        model, _, _, _ = cv_models
        predictions = model.predict(domain_transfer_test)
        for m in metrics:
            if m.name not in metrics_results:
                metrics_results[m.name] = []
            result = m(bioma_transfer_test, predictions)
            m.reset_states()
            metrics_results[m.name].append(result.numpy())
        predictions = tf.nn.softmax(predictions)
        all_predictions.append(predictions)
        # otus error
        se = tf.math.squared_difference(predictions, data_bioma_test_transformed)
        mse = tf.reduce_mean(se, axis=0)
        otus_errors.append(mse)
    mse_otus = tf.reduce_mean(tf.stack(otus_errors, axis=0), axis=0)
    mse_otus_keys = sorted(zip(mse_otus.numpy(), otu_columns), key=lambda x: x[0])
    for k, v in list(metrics_results.items()):
        v = np.asarray(v)
        metrics_results[k] = (v.mean(), v.min(), v.max())
    
    md_text = "## Test results \n"
    md_text += "| Metric           | Mean    | Min     | Max     |\n"
    md_text += "|:-----------------|--------:|--------:|--------:|\n"
    for k, v in metrics_results.items():
        md_text += "| {} | {} | {} | {} |\n".format(k, v[0], v[1], v[2])


    display(Markdown(md_text))

#     md_text ="### Best Otus\n"
#     md_text += "| OTU | mse |\n"
#     md_text += "|:----|----:|\n"
#     for v, k in mse_otus_keys[:10]:
#         md_text += "| {} | {} |\n".format(k, v)
#     md_text += "\n\n"
#     md_text +="### Worst Otus\n"
#     md_text += "| OTU | mse |\n"
#     md_text += "|:----|----:|\n"
#     for v, k in reversed(mse_otus_keys[-10:]):
#         md_text += "| {} | {} |\n".format(k, v)

#     display(Markdown(md_text))
    
    final_predictions = np.mean(all_predictions,axis=0)
    return final_predictions



def test_model_cv_predictions(models_cv, input_transform, output_transform, otu_columns, data_microbioma, data_domain):
    data_bioma_test_transformed = Percentage()(data_microbioma)
    if input_transform is not None:
        input_transform = input_transform()
    if output_transform is not None:
        output_transform = output_transform()
    metrics_results = {}
    metrics = get_experiment_metrics(input_transform, output_transform)[0][3:]
    otus_errors = []
    all_predictions = []
    for model in models_cv:
        predictions_latent = model[2].predict(data_domain)
        predictions = model[3].predict(predictions_latent)
        all_predictions.append(predictions)
    final_decoded = np.mean(all_predictions,axis=0)
    
    predictions = tf.nn.softmax(final_decoded)
    
    for m in metrics:
        if m.name not in metrics_results:
            metrics_results[m.name] = []
        result = m(data_microbioma, final_decoded)
        metrics_results[m.name] =result.numpy()
    # otus error
    se = tf.math.squared_difference(final_decoded, data_bioma_test_transformed)
    mse_otus = tf.reduce_mean(se, axis=0)
    mse_otus_keys = sorted(zip(mse_otus.numpy(), otu_columns), key=lambda x: x[0])
    for k, v in list(metrics_results.items()):
        v = np.asarray(v)
        metrics_results[k] = (v.mean(), v.min(), v.max())
    
    md_text = "## Test results \n"
    md_text += "| Metric           | Mean    | Min     | Max     |\n"
    md_text += "|:-----------------|--------:|--------:|--------:|\n"
    for k, v in metrics_results.items():
        md_text += "| {} | {} | {} | {} |\n".format(k, v[0], v[1], v[2])


    display(Markdown(md_text))

#     md_text ="### Best Otus\n"
#     md_text += "| OTU | mse |\n"
#     md_text += "|:----|----:|\n"
#     for v, k in mse_otus_keys[:10]:
#         md_text += "| {} | {} |\n".format(k, v)
#     md_text += "\n\n"
#     md_text +="### Worst Otus\n"
#     md_text += "| OTU | mse |\n"
#     md_text += "|:----|----:|\n"
#     for v, k in reversed(mse_otus_keys[-10:]):
#         md_text += "| {} | {} |\n".format(k, v)
#
#    display(Markdown(md_text))
    
    return predictions


def train_tl_noEnsemble(model_fn,
          data_latent_train,
          data_latent_val,
          data_domain_train,
          data_domain_val,
          epochs=100,
          batch_size=16,
          random_seed=347,
          verbose=0):
    train_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=epochs + 1, restore_best_weights=True)]
    if verbose >= 0:
        train_callbacks += [TqdmCallback(verbose=verbose)]

    tf.random.set_seed(random_seed)

    y_train, y_val = data_latent_train, data_latent_val
    x_train, x_val = data_domain_train, data_domain_val
    model = model_fn()
    metrics_prefix = 'domain'

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(
            batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        
    r = model.fit(train_dataset,
                  epochs=epochs,
                  validation_data=val_dataset,
                  callbacks=train_callbacks,
                  verbose=0)
    if metrics_prefix is not None:
        old_keys = r.history
        r.history = {}
        for k, v in old_keys.items():
            if k == 'loss' or k == 'val_loss':
                new_key = k
            elif k.startswith('val_'):
                new_key = 'val_{}_{}'.format(metrics_prefix, k[4:])
            else:
                new_key = '{}_{}'.format(metrics_prefix, k)
            r.history[new_key] = v
    return r, model


def test_model_tl_latent(model, latent_transfer_test, domain_transfer_test):

    metrics_results = {}

    final_predictions = model.predict(domain_transfer_test)
    
    result = se = tf.math.squared_difference(final_predictions, latent_transfer_test)
    metrics_results['mse'] =result.numpy()

    for k, v in list(metrics_results.items()):
        v = np.asarray(v)
        metrics_results[k] = (v.mean(), v.min(), v.max())
    
    md_text = "## Test results \n"
    md_text += "| Metric           | Mean    | Min     | Max     |\n"
    md_text += "|:-----------------|--------:|--------:|--------:|\n"
    for k, v in metrics_results.items():
       md_text += "| {} | {} | {} | {} |\n".format(k, v[0], v[1], v[2])
    
    #display(Markdown(md_text))
    
    return final_predictions


def test_model_tl_noEnsemble_FI(model, decoder, input_transform, output_transform, otu_columns, bioma_transfer_test, domain_transfer_test):
    data_bioma_test_transformed = Percentage()(bioma_transfer_test)
    if input_transform is not None:
        input_transform = input_transform()
    if output_transform is not None:
        output_transform = output_transform()
    metrics_results = {}
    metrics = get_experiment_metrics(input_transform, output_transform)[0][3:]
    otus_errors = []
    final_predictions = model.predict(domain_transfer_test)
    
    final_decoded = decoder.predict(final_predictions)
    
    predictions = tf.nn.softmax(final_decoded)

    for m in metrics:
        if m.name not in metrics_results:
            metrics_results[m.name] = []
        result = m(bioma_transfer_test, final_decoded)
        metrics_results[m.name] =result.numpy()
    # otus error
    se = tf.math.squared_difference(final_decoded, data_bioma_test_transformed)
    mse_otus = tf.reduce_mean(se, axis=0)
    mse_otus_keys = sorted(zip(mse_otus.numpy(), otu_columns), key=lambda x: x[0])
    for k, v in list(metrics_results.items()):
        v = np.asarray(v)
        metrics_results[k] = (v.mean(), v.min(), v.max())
    
    md_text = "## Test results \n"
    md_text += "| Metric           | Mean    | Min     | Max     |\n"
    md_text += "|:-----------------|--------:|--------:|--------:|\n"
    for k, v in metrics_results.items():
        md_text += "| {} | {} | {} | {} |\n".format(k, v[0], v[1], v[2])

    #display(Markdown(md_text))
    
    return metrics_results, predictions#esto predictions son los que estan relativizados a 1 (para que la suma de todos los OTUs sea 1)
