from graph import GraphLoader
from wtagnn import WTAGNN
import tensorflow as tf
from tensorflow import keras
import time
from eval import evaluate

def start_train(args):

    gloader = GraphLoader()
    g, nf, ef, e_label, train_mask, test_mask, val_mask = gloader.load_graph(args)  
    n_classes = 2
    n_edges = g.number_of_edges()
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    model = WTAGNN(g, input_node_feat_size, input_edge_feat_size, 
                    args.n_hidden, n_classes, args.n_layers,  tf.nn.relu, args.dropout)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
    dur = []

    print('Start train')
    for epoch in range(args.n_epochs):
        print('epoch: ', epoch)
        if epoch >= 3:
            t0 = time.time()
        
        with tf.GradientTape() as tape:
            n_logits, e_logits = model(nf,ef)
            loss = loss_fcn(e_label[train_mask], e_logits[train_mask])
            
            for weight in model.trainable_weights:
                loss = loss + args.weight_decay*tf.nn.l2_loss(weight)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc = evaluate(model, nf, ef, e_label, test_mask)
        
        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Accuracy: {}\nLoss: {}".format(acc, loss.numpy().item()))
        

