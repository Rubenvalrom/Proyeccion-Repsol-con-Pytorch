def entrenar_modelo(model, train_dataset, X_tensor_val, y_tensor_val, epochs=250, batch_size=64, learning_rate=0.03, scheduler=True, paciencia_scheduler=10, early_stopping=True, paciencia_early_stopping=30, show=True):
    """
    Entrena el modelo GRU con los datos de entrenamiento.
    
    Parámetros:
    - model (nn.Module): Instancia del modelo GRU.
    - train_dataset (TensorDataset): Dataset de entrenamiento.
    - X_tensor_val (Tensor): Tensor de características de validación.
    - y_tensor_val (Tensor): Tensor de etiquetas de validación.
    - epochs (int): Número de épocas para entrenar.
    - batch_size (int): Tamaño del lote para el entrenamiento.
    - learning_rate (float): Tasa de aprendizaje para el optimizador.
    - scheduler (bool): Indica si se debe utilizar un programador de tasa de aprendizaje.
    - paciencia_scheduler (int): Número de épocas para esperar antes de reducir la tasa de aprendizaje.
    - early_stopping (bool): Indica si se debe utilizar la detención temprana.
    - paciencia_early_stopping (int): Número de épocas para esperar antes de detener el entrenamiento.
    - show (bool): Indica si se deben mostrar los mensajes de entrenamiento.

    Devuelve:
    - model (nn.Module): Modelo entrenado.
    """
    torch.manual_seed(27)
    
    model = model.to(device)
    X_tensor_val = X_tensor_val.to(device)
    y_tensor_val = y_tensor_val.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,)
    criterion = nn.L1Loss()
    if scheduler:
        step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=paciencia_scheduler, gamma=0.75)

    if early_stopping:
        mejor_perdida = float('inf')
        epochs_sin_mejorar = 0
        mejor_modelo = None

    generador = torch.Generator()
    generador.manual_seed(27)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generador)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        if scheduler:
            step_lr.step()

        if early_stopping:
            with torch.no_grad():
                model.eval()
                val_outputs = model(X_tensor_val)
                val_loss = criterion(val_outputs.squeeze(), y_tensor_val)
           
            if val_loss.item() < mejor_perdida:
                mejor_perdida = val_loss.item()
                epochs_sin_mejorar = 0
                mejor_modelo = copy.deepcopy(model.state_dict())
            else:
                epochs_sin_mejorar += 1

                if epochs_sin_mejorar >= paciencia_early_stopping:
                    if show:
                        print(f'No se observó mejora en {paciencia_early_stopping} épocas. Deteniendo entrenamiento.')

                    model.load_state_dict(mejor_modelo)
                    return model

        if show and ((epoch + 1) % 10 == 0):
            print(f'Época [{epoch + 1}/{epochs}], Pérdida: {loss.item():.4f}')
            if early_stopping:
                print(f'Mejor pérdida de validación actual:{mejor_perdida:.4f}')
    if early_stopping and mejor_modelo:
        model.load_state_dict(mejor_modelo)
    return model


def calcular_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    return mae

def calcular_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_pred - y_true) / y_true) * 100)
    return mape

def evaluar_modelo(model, X_seq_tensor, y, scaler=None, titulo='Evaluación del Modelo', show=True):
    """
    Evalúa el modelo entrenado calculando métricas y mostrando gráficos.

    Parámetros:
    - model (nn.Module): Modelo entrenado de PyTorch.
    - X_seq_tensor (torch.Tensor): Tensor con las secuencias de entrada (shape: [N, seq_len, n_features]).
    - y (np.ndarray | torch.Tensor): Valores reales de la variable objetivo.
    - scaler (object, opcional): Escalador (por ejemplo StandardScaler) usado para desescalar y_pred y y. Default: None.
    - titulo (str): Título que se mostrará en la figura. Default: 'Evaluación del Modelo'.
    - show (bool): Si True imprime métricas y muestra los gráficos. Default: True.

    Devuelve:
    - mae (float): Mean Absolute Error entre valores reales y predichos (en la escala original si se proporciona scaler).
    - mape (float): Mean Absolute Percentage Error (%) (en la escala original si se proporciona scaler).
    """
    
    with torch.inference_mode():
        model.eval()
        model = model.to(device)
        y_pred = model(X_seq_tensor).squeeze().cpu().numpy()

    if type(y) is torch.Tensor:
        y = y.cpu().numpy()

    if scaler:
        y_pred = scaler.inverse_transform([y_pred]).flatten()
        y = scaler.inverse_transform([y]).flatten()

    mae = calcular_mae(y, y_pred)
    mape = calcular_mape(y, y_pred)

    if show:

        print(f'MAE: {mae:.4f}')
        print(f'MAPE: {mape:.4f}%')
        
        fig, axs = plt.subplots(3, 1, figsize=(14, 14))

        # Gráfico de líneas: Real vs Predicción
        axs[0].plot(y, label='Real', color="#1C9DE2", lw=0.75)
        axs[0].plot(y_pred, label='Predicción', color="#ff1414", alpha=0.5, lw=0.75)
        axs[0].set_title('Análisis Cronológico de la Precisión del Modelo')
        axs[0].set_xlabel('Fecha')
        axs[0].set_ylabel('Valor de la variable objetivo')
        axs[0].legend()

        # Gráfico de dispersión: Predicciones vs Valores Reales
        sns.scatterplot(x=y, y=y_pred, color='#1C9DE2', alpha=0.25, ax=axs[1], size=3, legend=False)
        axs[1].plot([y.min(), y.max()], [y.min(), y.max()], color='#ff1414', linestyle='--')
        axs[1].set_title('Desempeño Global del Modelo: Valores Reales vs Estimados')
        axs[1].set_xlim(y.min(), y.max())
        axs[1].set_ylim(y.min(), y.max())
        axs[1].set_xlabel('Valores Reales')
        axs[1].set_ylabel('Predicciones')

        # Histograma del error porcentual
        sns.histplot( (y_pred - y) / y * 100, bins=100, color="#aa00ff", ax=axs[2], linewidth=1.5)
        axs[2].set_title('Distribución del Error Porcentual')
        axs[2].set_xlabel('Error Porcentual (%)')
        axs[2].set_ylabel('Frecuencia')

        plt.tight_layout()
        plt.suptitle(titulo, fontsize=16, y=1.02)
        plt.show()


    return mae, mape