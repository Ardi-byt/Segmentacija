import cv2 as cv
import numpy as np

def gaussovo_jedro(d,h):
    #Enacba za izracun jedra po formuli 
    return np.exp(-d**2 / (2* h**2))

def kmeans(slika, k=3, iteracije=10): #Koncana K-means funkcija
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    visina, sirina, _ = slika.shape #Dobimo dimenzije slike
    slika = slika.astype(np.float32)
    
    # Pridobi vhodne parametre
    izbira = input("Vnesi način izbire centrov ('rocno' ali 'avto'): ").strip()
    dimenzija = int(input("Vnesi dimenzijo (3 za barve, 5 za barve+lokacija): "))
    T = int(input("Vnesi minimalno razdaljo T (npr. 30): "))
    
    # Priprava prostora znacilnic
    if dimenzija == 3:
        piksli = slika.reshape(-1, 3)
    elif dimenzija == 5:
        y, x = np.mgrid[0:visina, 0:sirina]
        # Normaliziraj koordinate
        lokacije = np.stack((x.flatten()/sirina, y.flatten()/visina), axis=1)
        barve = slika.reshape(-1, 3)
        piksli = np.hstack((barve, lokacije)).astype(np.float32)
    
    # Izračun začetnih centrov
    centri = izracunaj_centre(slika, izbira, dimenzija, T)
    
    # Glavna zanka K-means
    for _ in range(iteracije):
        # Izračunaj razdalje med piksli in centri (Evklidska razdalja)
        razdalje = np.linalg.norm(piksli[:, None] - centri, axis=2)
        # Dodeli piksle najbližjim centrom
        oznake = np.argmin(razdalje, axis=1)
        
        # Posodobi centre
        novi_centri = []
        for i in range(len(centri)):
            cluster = piksli[oznake == i]
            if len(cluster) > 0:
                novi_centri.append(cluster.mean(axis=0))
        centri = np.array(novi_centri)
    
    # Ustvari segmentirano sliko
    segmentirana = centri[oznake][:, :3].reshape(visina, sirina, 3).astype(np.uint8)
    return segmentirana


def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass


def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    visina, sirina, _ = slika.shape #Dobimo dimenzije slike

    #Priprava prostora znacilnic
    if dimenzija_centra == 3:
        #3D uporabimo samo BGR
        znacilnice = slika.reshape(-1, 3).astype(np.float32)
    elif dimenzija_centra == 5:
        #5D uporabimo BGR in se x pa y
        y, x = np.mgrid[0:visina, 0:sirina] #Ustvari mrezo koordinat
        #Normalizcija koordinat
        lokacije = np.stack((x.flatten()/sirina, y.flatten()/visina), axis=1)
        barve = slika.reshape(-1, 3)
        znacilnice = np.hstack((barve, lokacije)).astype(np.float32)
        #Zdruzi barve in lokacije
    else:
        raise ValueError("dimenzija_centra mora biti 3 ali 5")

    centri = []
    st_centrov = 3  # Vedno izberemo 3 centre, ker uporabljamo samo 3 centre pri kmeans funkciji
    
    if izbira == "rocno":
        # Zagotovimo, da uporabljamo BGR za prikaz (OpenCV privzeto)
        slika_prikaz = slika.copy().astype(np.uint8)
        print(f"Kliknite na sliko, da izberete {st_centrov} centrov (ESC za konec).")

        def klik(event, x, y, flags, param):
            #Funkcija ki se sprozi ob kliku z misko
            if event == cv.EVENT_LBUTTONDOWN and len(centri) < st_centrov:
                if dimenzija_centra == 3:
                    #Za 3D vzame barvo izbranega piksla
                    center = slika[y, x].astype(np.float32)
                else:
                    #Za 5D vzame barvo in lokacijo piksla
                    barva = slika[y, x].astype(np.float32)
                    lokacija = np.array([x/sirina, y/visina], dtype=np.float32)
                    center = np.concatenate((barva, lokacija))
                centri.append(center) #Center v seznamu
                print(f"Izbran center {len(centri)}: {center}")
                cv.circle(slika_prikaz, (x, y), 5, (0, 255, 0), -1)
                cv.imshow("Izberi centre", slika_prikaz)
        #Nastavi okno in callback za klike
        cv.namedWindow("Izberi centre")
        cv.setMouseCallback("Izberi centre", klik)
        #Prikazuje sliko dokler niso izbrani vsi centri ali stisnemo ESC
        while len(centri) < st_centrov:
            cv.imshow("Izberi centre", slika_prikaz)
            if cv.waitKey(1) == 27:  # ESC
                break
        cv.destroyAllWindows()

    #Avtomatska izbira z upoštevanjem praga T
    elif izbira == "avto":
        st_vzorcev = znacilnice.shape[0] #Stevilo vseh kandidatov
        rng = np.random.default_rng() #Generator nakljucnih stevil
        poskusi = 0
        max_poskusov = 1000 #Prepreci neskoncno zanko

        while len(centri) < st_centrov and poskusi < max_poskusov:
            #Nakljucno izberi center
            kandidat = znacilnice[rng.integers(0, st_vzorcev)]
            #Preveri ali je dovolj oddaljen od ze izbranih centrov
            if all(np.linalg.norm(kandidat - c) > T for c in centri):
                centri.append(kandidat)
            poskusi += 1

        if len(centri) < st_centrov:
            raise RuntimeError(f"Ne najdem {st_centrov} centrov z T={T}!")

    else:
        raise ValueError("Neveljavna izbira. Dovoljeno: 'rocno' ali 'avto'")

    #Vrni center kot numpy array
    return np.array(centri, dtype=np.float32)



if __name__ == "__main__":
    # Nastavitve za lepši izpis centrov
    np.set_printoptions(suppress=True, precision=4)
    
    # Naloži sliko v BGR formatu (OpenCV privzeto)
    slika = cv.imread("./.utils/zelenjava.jpg")
    if slika is None:
        print("Napaka: Slika ni bila najdena!")
        exit()
    
    # Izvedi K-means segmentacijo
    segmentirana = kmeans(slika, k=3, iteracije=10)
    
    # Prikaži originalno in segmentirano sliko
    cv.imshow("Original", slika)
    cv.imshow("Segmentirana", segmentirana)
    cv.waitKey(0)
    cv.destroyAllWindows()

