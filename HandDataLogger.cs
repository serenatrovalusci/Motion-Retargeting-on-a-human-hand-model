using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class HandDataLogger : MonoBehaviour
{
    private string filePath;
    private StreamWriter writer;

    // Timer per la registrazione ogni 3 secondi
    private float lastSaveTime = 0f;
    private float saveInterval = 3f; // Tempo tra una registrazione e l'altra

    // Riferimento ai joint di Weart
    public Transform thumb1;
    public Transform thumb2;
    public Transform thumb3;

    public Transform index1;
    public Transform index2;
    public Transform index3;

    public Transform middle1;
    public Transform middle2;
    public Transform middle3;

    public float closure;

    // Start is called before the first frame update
    void Start()
    {
        filePath = Application.persistentDataPath + "/hand_dataset.csv";
        
        bool fileExists = File.Exists(filePath);  // Controllo PRIMA se il file esiste

        // Apro il file solo DOPO aver controllato se esiste
        writer = new StreamWriter(filePath, true);

        if (!fileExists)
        {
            writer.WriteLine("Joint1Tx,Joint1Ty,Joint1Tz,Joint2Tx,Joint2Ty,Joint2Tz,Joint3Tx,Joint3Ty,Joint3Tz,Joint1Ix,Joint1Iy,Joint1Iz, Joint2Ix,Joint2Iy, Joint2Iz, Joint3Ix, Joint3Iy, Joint3Iz,Joint1Mx,Joint1My,Joint1Mz, Joint2Mx,Joint2My, Joint2Mz, Joint3Mx, Joint3My, Joint3Mz,Closure");
            writer.Flush();
        }

        // Trova i joint nel prefab usando i nomi precisi
        thumb1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R");
        thumb2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R");
        thumb3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R/DEF-thumb.03.R");

        index1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R");
        index2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R");
        index3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R/DEF-f_index.03.R");

        middle1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R");
        middle2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R");
        middle3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R/DEF-f_middle.03.R");

        // Check if all joints were found
    if (thumb1 == null) Debug.LogError(" Missing Joint: DEF-thumb.01.R");
    if (thumb2 == null) Debug.LogError(" Missing Joint: DEF-thumb.02.R");
    if (thumb3 == null) Debug.LogError(" Missing Joint: DEF-thumb.03.R");

    if (index1 == null) Debug.LogError(" Missing Joint: DEF-f_index.01.R");
    if (index2 == null) Debug.LogError(" Missing Joint: DEF-f_index.02.R");
    if (index3 == null) Debug.LogError(" Missing Joint: DEF-f_index.03.R");

    if (middle1 == null) Debug.LogError(" Missing Joint: DEF-f_middle.01.R");
    if (middle2 == null) Debug.LogError(" Missing Joint: DEF-f_middle.02.R");
    if (middle3 == null) Debug.LogError(" Missing Joint: DEF-f_middle.03.R");

    Debug.Log(" Joint search complete.");
    }

    void Update()
    {
        // Controlla se sono passati almeno 3 secondi dall'ultima registrazione
        if (Time.time - lastSaveTime >= saveInterval)
        {
            lastSaveTime = Time.time; // Aggiorna il tempo dell'ultima registrazione

            if (thumb1 != null && thumb2 != null && thumb3 != null &&
                index1 != null && index2 != null && index3 != null &&
                middle1 != null && middle2 != null && middle3 != null)
            {
                // Ottieni le rotazioni dei joint
                Vector3 Joint1T = thumb1.localEulerAngles;
                Vector3 Joint2T = thumb2.localEulerAngles;
                Vector3 Joint3T = thumb3.localEulerAngles;

                Vector3 Joint1I = index1.localEulerAngles;
                Vector3 Joint2I = index2.localEulerAngles;
                Vector3 Joint3I = index3.localEulerAngles;

                Vector3 Joint1M = middle1.localEulerAngles;
                Vector3 Joint2M = middle2.localEulerAngles;
                Vector3 Joint3M = middle3.localEulerAngles;

                //Normalizzare gli angoli

                Joint1T = new Vector3(NormalizeAngle(Joint1T.x), NormalizeAngle(Joint1T.y), NormalizeAngle(Joint1T.z));
                Joint2T = new Vector3(NormalizeAngle(Joint2T.x), NormalizeAngle(Joint2T.y), NormalizeAngle(Joint2T.z));
                Joint3T = new Vector3(NormalizeAngle(Joint3T.x), NormalizeAngle(Joint3T.y), NormalizeAngle(Joint3T.z));

                Joint1I = new Vector3(NormalizeAngle(Joint1I.x), NormalizeAngle(Joint1I.y), NormalizeAngle(Joint1I.z));
                Joint2I = new Vector3(NormalizeAngle(Joint2I.x), NormalizeAngle(Joint2I.y), NormalizeAngle(Joint2I.z));
                Joint3I = new Vector3(NormalizeAngle(Joint3I.x), NormalizeAngle(Joint3I.y), NormalizeAngle(Joint3I.z));

                Joint1M = new Vector3(NormalizeAngle(Joint1M.x), NormalizeAngle(Joint1M.y), NormalizeAngle(Joint1M.z));
                Joint2M = new Vector3(NormalizeAngle(Joint2M.x), NormalizeAngle(Joint2M.y), NormalizeAngle(Joint2M.z));
                Joint3M = new Vector3(NormalizeAngle(Joint3M.x), NormalizeAngle(Joint3M.y), NormalizeAngle(Joint3M.z));

                // Registra i dati solo se `writer` non è null
                if (writer != null)
                {
                    Debug.Log(" Scrivendo nel file CSV..."); // Messaggio di debug
                    LogData(Joint1T, Joint2T, Joint3T, Joint1I, Joint2I, Joint3I, Joint1M, Joint2M, Joint3M, closure);
                    Debug.Log($"Joint1T:  { Joint1T.x}");

                }
            }
        }
    }

    float NormalizeAngle(float angle)
    {
        return (angle > 180f) ? angle - 360f : angle;
    }

    public void LogData(Vector3 Joint1T, Vector3 Joint2T, Vector3 Joint3T, Vector3 Joint1I, Vector3 Joint2I, Vector3 Joint3I, Vector3 Joint1M, Vector3 Joint2M, Vector3 Joint3M, float closure)
    {
        if (writer != null)  // Controllo che `writer` sia valido
        {
            string line = Joint1T.x + "," + Joint1T.y + "," + Joint1T.z + "," +
              Joint2T.x + "," + Joint2T.y + "," + Joint2T.z + "," +
              Joint3T.x + "," + Joint3T.y + "," + Joint3T.z + "," +
              Joint1I.x + "," + Joint1I.y + "," + Joint1I.z + "," +
              Joint2I.x + "," + Joint2I.y + "," + Joint2I.z + "," +
              Joint3I.x + "," + Joint3I.y + "," + Joint3I.z + "," +
              Joint1M.x + "," + Joint1M.y + "," + Joint1M.z + "," +
              Joint2M.x + "," + Joint2M.y + "," + Joint2M.z + "," +
              Joint3M.x + "," + Joint3M.y + "," + Joint3M.z + "," +
              closure;

            writer.WriteLine(line);
            writer.Flush();
            Debug.Log(" Dati salvati nel file CSV!"); // Debug per confermare il salvataggio
        }
        else
        {
            Debug.LogError(" Errore: writer è null, impossibile scrivere nel file.");
        }
    }

    void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Close();
            writer.Dispose();
            Debug.Log(" File CSV chiuso correttamente!");
        }
    }
}
