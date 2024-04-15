function guardarEspectrograma(signal,Fs,nombre)
    nombre = "datasetTotal/"+nombre;
    figure;
    plot(real(signal))

    % Convierte el gráfico a blanco y negro
    colormap('gray');

    % Establece el tamaño de la figura a 224x224 píxeles
    set(gcf, 'Position',  [100, 100, 224, 224]);

    axis off
    print(gcf,nombre,'-dpng','-r300'); 
    %saveas(fig,nombre)
    close all
    % Guarda la imagen capturada
    %imwrite(espectrogama, nombre);


end

