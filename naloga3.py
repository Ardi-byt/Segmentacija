import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    pass

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass


def izracunaj_centre(slika, izbira, dimenzija_centra=3, T=30):
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
    if izbira == "rocno":
        slika_prikaz = slika.copy()
        print(f"Kliknite na sliko, da izberete {dimenzija_centra} centrov (ESC za konec).")

        def klik(event, x, y, flags, param):
            #Funkcija ki se sprozi ob kliku z misko
            if event == cv.EVENT_LBUTTONDOWN and len(centri) < dimenzija_centra:
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
        while len(centri) < dimenzija_centra:
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

        while len(centri) < dimenzija_centra and poskusi < max_poskusov:
            #Nakljucno izberi center
            kandidat = znacilnice[rng.integers(0, st_vzorcev)]
            #Preveri ali je dovolj oddaljen od ze izbranih centrov
            if all(np.linalg.norm(kandidat - c) > T for c in centri):
                centri.append(kandidat)
            poskusi += 1

        if len(centri) < dimenzija_centra:
            raise RuntimeError(f"Ne najdem {dimenzija_centra} centrov z T={T}!")

    else:
        raise ValueError("Neveljavna izbira. Dovoljeno: 'rocno' ali 'avto'")

    #Vrni center kot numpy array
    return np.array(centri, dtype=np.float32)

if __name__ == "__main__":
    # Nalozi sliko v BGR (OpenCV privzeto)
    slika = cv.imread("./.utils/zelenjava.jpg")
    if slika is None:
        print("Napaka: Slika ni bila najdena!")
        exit()

    # Uporabniški vnos
    izbira = input("Vnesi način izbire centrov ('rocno' ali 'avto'): ").strip()
    dimenzija = int(input("Vnesi dimenzijo (3 za barve, 5 za barve+lokacija): "))
    T = int(input("Vnesi minimalno razdaljo T (npr. 30): "))

    #Izracuna centre
    centri = izracunaj_centre(slika, izbira, dimenzija, T)
    np.set_printoptions(suppress=True)
    print(f"Izbrani centri ({dimenzija}D):\n", centri)
